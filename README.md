# SAM2 Nuclio ML Backend for Label Studio

SAM2-Nuclio provides integration between Meta’s Segment Anything 2 (SAM2) model with Nuclio and Label Studio using Label Stuido ML backend SDK. The code is largely based on the [SAM2 example from HumanSignal's Label Studio ML backend](https://github.com/HumanSignal/label-studio-ml-backend/blob/master/label_studio_ml/examples/segment_anything_2_image/model.py) and is released under the Apache License, Version 2.0.

> **Note:** This project inherits and adapts code from the original SAM2 example to run as a serverless Nuclio function. All modifications are solely for integration with Nuclio. For full license details, see the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Overview

The Nuclio function exposes the following endpoints:
- **/setup:** Initializes the model with a label configuration.
- **/predict:** Processes prediction requests from Label Studio.
- **/webhook:** Processes training events (if applicable).
- **/health:** Returns the health status of the ML backend.
- **/live** Returns a liveness check.
- **/train** Currently does nothing; reserved for the future improvements.

The SAM2 model is configured to use a specific configuration file and checkpoint, and a dummy label interface is provided to supply default behavior for required label operations for Label Studio.

## How the Prediction Logic is Implemented

To successfully integrate the Label Studio ML backend with Nuclio, I implemented the predict method following Label Studio's expectations:

- `tasks`: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
- `context`: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Support-interactive-pre-annotations-in-your-ML-backend) - for
  interactive labeling scenario
- `predictions`: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Raw-JSON-format-of-completed-tasks)

In addition, I implemented `DummyLabelInterface` to provide a minimal implementation required by the base `class LabelStudioMLBase`. It ensures the model does not fail if a complete label configuration is not provided during initialization by supplying default tag occurrences:

```python
class DummyLabelInterface:
    def get_first_tag_occurrence(self, from_name="BrushLabels", to_name="Image", **kwargs):
        return from_name, to_name, "image"
```
### Sam2Nuclio Model

The Sam2Nuclio model inherits from LabelStudioMLBase and is specifically designed for deployment with Nuclio.

The predict method works as follows:

* Extracts context information provided by Label Studio, including keypoints (keypointlabels) and rectangles (rectanglelabels) from user annotations.
* Processes keypoints and rectangle annotations to calculate point coordinates and bounding boxes.
* Retrieves the image URL from the provided tasks and loads the image using a helper method.
* Performs segmentation predictions using the SAM2 model based on the provided prompts (points and bounding boxes).
* Generates segmentation masks and returns them along with associated prediction probabilities in Label Studio's expected JSON format.

See [Label Studio ML Backend Documentation] (https://labelstud.io/guide/ml_create) for more details.

## Environment Variables

Configure your Nuclio function with the following environment variables:

- `SEGMENT_ANYTHING_2_REPO_PATH`: `/opt/nuclio/segment-anything-2`
- `ROOT_DIR`: `/opt/nuclio`
- `MODEL_CONFIG`: `sam2_hiera_l.yaml`  
  *Ensure that the configuration file is available at `/opt/nuclio/segment-anything-2/sam2_configs/sam2_hiera_l.yaml` or copied to the expected location.*
- `MODEL_CHECKPOINT`: `/opt/nuclio/segment-anything-2/checkpoints/sam2.1_hiera_large.pt`
- `DEVICE`: `cuda`
- `PYTHONPATH`: `/opt/nuclio/segment-anything-2`
- `LABEL_STUDIO_HOST`: `https://<YOUR_LABEL_STUDIO_HOST>:<PORT>`
- `LABEL_STUDIO_API_KEY`: `<YOUR_PERSONAL_ACCESS_TOKEN>` *(legacy `LABEL_STUDIO_ACCESS_TOKEN` still works but won't refresh automatically)*
- `LOG_LEVEL`: `DEBUG` *(Set to INFO in production)*

`LABEL_STUDIO_HOST` and `LABEL_STUDIO_API_KEY` are only needed if you want to download images from the Label Studio instance to the SAM ML Backend.

### Authenticating to Label Studio

Label Studio Personal Access Tokens (PATs) behave differently from legacy tokens: they are JWT refresh tokens and cannot be sent directly to endpoints such as `/tasks/<id>/presign`. This backend now exchanges your PAT (provided via `LABEL_STUDIO_API_KEY`) for a short-lived bearer token using `POST /api/token/refresh`, caches it, and refreshes it automatically before it expires (~5 minutes). The bearer token is then passed to the Label Studio SDK helper so downloading task media keeps working even when PATs are mandatory.

If you still rely on legacy tokens, set `LABEL_STUDIO_ACCESS_TOKEN` instead—they will be forwarded unchanged, but they won't benefit from the refresh flow once PATs become the only option.

## Quickstart

### Container Image

Although Nuclio can build container images during deployment using Kaniko, this approach can slow down deployment significantly due to the relatively large size (~12GB) of the image. Therefore, I recommend building and pushing the Docker image separately outside of the deployment process.

A Dockerfile already exists in the repository. To build and push the container image to a private (or public) registry, replace registry.example.com/sam2-nuclio:v0.1 with your URL and desired image tag

```bash
docker build -t registry.example.com/sam2-nuclio:v0.1 .
docker push registry.example.com/sam2-nuclio:v0.1
```

or use my pre-built image (in your own sake) `armagankaratosun/sam2-nuclio:v0.64` and use that in your Nuclio function yaml manifest.

### Nuclio Deployment Instructions

Create a yaml manifest e.g, `sam2-nuclio.yaml` with the following content:

```yaml
apiVersion: nuclio.io/v1
kind: Function
metadata:
  name: sam2-nuclio
spec:
  runtime: python:3.10
  handler: handler:handler
  image: registry.example.com/sam2-nuclio:v0.1

  runtimeClassName: nvidia  

  env:
    - name: SEGMENT_ANYTHING_2_REPO_PATH
      value: "/opt/nuclio/segment-anything-2"
    - name: ROOT_DIR
      value: "/opt/nuclio"
    - name: MODEL_CONFIG
      value: "sam2_hiera_l.yaml"
    - name: MODEL_CHECKPOINT
      value: "/opt/nuclio/segment-anything-2/checkpoints/sam2_hiera_large.pt"
    - name: DEVICE
      value: "cuda"
    - name: PYTHONPATH
      value: "/opt/nuclio/segment-anything-2"
    - name: LABEL_STUDIO_HOST
      value: "https://<YOUR_LABEL_STUDIO_HOST>:<PORT>"
    - name: LABEL_STUDIO_API_KEY
      value: "<YOUR_PERSONAL_ACCESS_TOKEN>"
    - name: LOG_LEVEL
      value: INFO

  resources:
    limits:
      nvidia.com/gpu: 1  # Allocate 1 GPU for this function
    requests:
      cpu: 250m
      memory: 1Gi

  triggers:
    http:
      kind: http
      numWorkers: 1
      attributes:
        serviceType: ClusterIP
        port: 8080
```

Deploy your function using the Nuclio CLI:

```bash
nuctl deploy --project-name ml-backends --file sam2-nuclio.yaml --namespace <your-namespace> --platform kube
```

## Author
Armagan Karatosun

## References

- **Label Studio Documentation:**
  - [Task Format (JSON)](https://labelstud.io/guide/task_format.html)
  - [Interactive Pre-annotations](https://labelstud.io/guide/ml_create#Support-interactive-pre-annotations-in-your-ML-backend)
  - [Predictions JSON Format](https://labelstud.io/guide/export.html#Raw-JSON-format-of-completed-tasks)
  - [Creating ML Backend for Label Studio](https://labelstud.io/guide/ml_create)

- **HumanSignal Label Studio ML Backend Example:**
  - [Segment Anything 2 (SAM2) Example](https://github.com/HumanSignal/label-studio-ml-backend/blob/master/label_studio_ml/examples/segment_anything_2_image/model.py)
  - [ML Backend Prediction Logic](https://github.com/HumanSignal/label-studio-ml-backend/tree/master?tab=readme-ov-file#3-implement-prediction-logic)

- **Segment Anything Model 2 (Meta AI):**
  - [Segment Anything 2 Repository ](https://github.com/facebookresearch/sam2)

- **Nuclio Serverless Platform:**
  - [Nuclio Documentation](https://nuclio.io/)
  - [Nuclio GitHub Repository](https://github.com/nuclio/nuclio)
