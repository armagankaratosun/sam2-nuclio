import json
from model import Sam2Nuclio

model = None
TRAIN_EVENTS = ["train_start", "train_end"] # in progress. does nothing atm.

def init_model(context):
    global model
    if model is None:
        context.logger.info("Initializing SAM2 model...")
        model = Sam2Nuclio()
    else:
        context.logger.info("Model already initialized.")

def handler(context, event):
    global model
    path = event.path.lower()

    if path == "/train":
        context.logger.info("Training request received.")
        return handle_train(context, event)
    if path == "/setup":
        return handle_setup(context, event)
    elif path == "/predict":
        if model is None:
            init_model(context)
        context.logger.info("Received prediction request.")
        return handle_predict(context, event)
    elif path == "/webhook":
        context.logger.info("Webhook endpoint hit.")
        return handle_webhook(context, event)
    elif path in ["/", "/health", "/__internal/health"]:
        context.logger.info("Health check received.")
        health_response = {
            "status": "UP",
            "model_class": model.__class__.__name__ if model else "Not Initialized"
        }
        return context.Response(
            body=json.dumps(health_response),
            headers={"Content-Type": "application/json"},
            status_code=200
        )
    elif path == "/live":
        context.logger.info("Liveness probe hit.")
        return context.Response(
            body=json.dumps({"status": "live"}),
            headers={"Content-Type": "application/json"},
            status_code=200
        )
    else:
        context.logger.info("Unhandled path: " + path)
        return context.Response(
            body=json.dumps({"error": "Not Found"}),
            headers={"Content-Type": "application/json"},
            status_code=404
        )

def handle_predict(context, event):
    # Log headers at INFO level (can comment out if too verbose)
    context.logger.info("Received headers: " + str(event.headers))
    context.logger.info("Entered handle_predict")
    
    if isinstance(event.body, bytes):
        body_str = event.body.decode("utf-8")
    elif isinstance(event.body, dict):
        body_str = json.dumps(event.body)
    else:
        body_str = event.body

    try:
        data = json.loads(body_str)
        # context.logger.info(f"Parsed payload: {data}")
    except Exception as e:
        context.logger.error("Failed to parse request body: " + str(e))
        return context.Response(
            body=json.dumps({"error": "Invalid JSON in request"}),
            headers={"Content-Type": "application/json"},
            status_code=400
        )

    tasks = data.get("tasks", data)

    # Extract context from payload 'params' if available; fallback to header.
    if isinstance(data, dict) and "params" in data and isinstance(data["params"], dict) and "context" in data["params"]:
        context_data = data["params"]["context"]
        # context.logger.info(f"Extracted context from payload: {context_data}")
    else:
        context_header = (
            event.headers.get("x-label-studio-context") or
            event.headers.get("ls-context") or
            "{}"
        )
        try:
            context_data = json.loads(context_header)
            # context.logger.info(f"Parsed context from header: {context_data}")
        except Exception as e:
            context.logger.error("Failed to parse context header: " + str(e))
            context_data = {}

    try:
        model_response = model.predict(tasks, context=context_data)
        context.logger.info(f"Raw model response: {model_response}")
        context.logger.info("Model.predict executed successfully")
    except Exception as e:
        context.logger.error("Error during prediction: " + str(e))
        return context.Response(
            body=json.dumps({"error": str(e)}),
            headers={"Content-Type": "application/json"},
            status_code=500
        )

    try:
        if hasattr(model_response, "has_model_version") and callable(model_response.has_model_version):
            if not model_response.has_model_version():
                mv = model.get("model_version") or "v1.0"
                model_response.set_version(mv)
                context.logger.info(f"Model version set to: {mv}")
            else:
                model_response.update_predictions_version()
    except Exception as e:
        context.logger.error("Error updating model version in response: " + str(e))

    try:
        response_dict = model_response.dict()
        # context.logger.info(f"ModelResponse dict: {response_dict}")
    except Exception as e:
        context.logger.error("Error serializing response: " + str(e))
        response_dict = {"error": "Serialization error"}

    final_response = {"results": response_dict.get("predictions", response_dict)}

    try:
        response_body = json.dumps(final_response, ensure_ascii=False)
    except Exception as e:
        context.logger.error("Error final serializing response: " + str(e))
        response_body = json.dumps({"error": "Final serialization error"}, ensure_ascii=False)

    context.logger.info("handle_predict finished, sending response")
    return context.Response(
        body=response_body,
        headers={"Content-Type": "application/json"},
        status_code=200,
    )

def handle_setup(context, event):
    if event.body is None:
        context.logger.error("No setup request body provided")
        return context.Response(
            body=json.dumps({"error": "No setup request body provided"}),
            headers={"Content-Type": "application/json"},
            status_code=400
        )
    if isinstance(event.body, bytes):
        body_str = event.body.decode("utf-8")
    elif isinstance(event.body, dict):
        body_str = json.dumps(event.body)
    else:
        body_str = event.body

    try:
        data = json.loads(body_str)
    except Exception as e:
        context.logger.error("Failed to parse setup request body: " + str(e))
        return context.Response(
            body=json.dumps({"error": "Invalid JSON in setup request"}),
            headers={"Content-Type": "application/json"},
            status_code=400
        )

    project = data.get("project")
    project_id = project.split(".", 1)[0] if project else ""
    label_config = data.get("schema")
    extra_params = data.get("extra_params")

    context.logger.info(f"Setup request received with project_id: {project_id}, label_config: {label_config}")

    try:
        new_model = Sam2Nuclio(project_id=project_id, label_config=label_config)
    except Exception as e:
        context.logger.error("Error initializing model in setup: " + str(e))
        return context.Response(
            body=json.dumps({"error": "Model initialization failed: " + str(e)}),
            headers={"Content-Type": "application/json"},
            status_code=500
        )
    
    if extra_params:
        try:
            new_model.set_extra_params(extra_params)
        except Exception as e:
            context.logger.error("Error setting extra params in setup: " + str(e))
    
    global model
    model = new_model

    try:
        model_version = model.get("model_version")
    except Exception as e:
        context.logger.error("Error retrieving model version: " + str(e))
        model_version = "unknown"

    response_data = {"model_version": model_version}
    return context.Response(
        body=json.dumps(response_data),
        headers={"Content-Type": "application/json"},
        status_code=200
    )

def handle_webhook(context, event):
    if event.body is None:
        context.logger.error("No webhook request body provided")
        return context.Response(
            body=json.dumps({"error": "No webhook request body provided"}),
            headers={"Content-Type": "application/json"},
            status_code=400
        )
    if isinstance(event.body, bytes):
        body_str = event.body.decode("utf-8")
    elif isinstance(event.body, dict):
        body_str = json.dumps(event.body)
    else:
        body_str = event.body

    try:
        data = json.loads(body_str)
    except Exception as e:
        context.logger.error("Failed to parse webhook request body: " + str(e))
        return context.Response(
            body=json.dumps({"error": "Invalid JSON in webhook request"}),
            headers={"Content-Type": "application/json"},
            status_code=400
        )

    action = data.pop("action", None)
    if action not in TRAIN_EVENTS:
        context.logger.info(f"Webhook event '{action}' not recognized")
        return context.Response(
            body=json.dumps({"status": "Unknown event"}),
            headers={"Content-Type": "application/json"},
            status_code=200
        )

    project = data.get("project", {})
    project_id = str(project.get("id", ""))
    label_config = project.get("label_config", "")
    context.logger.info(f"Webhook received for project_id: {project_id}")

    try:
        webhook_model = Sam2Nuclio(project_id=project_id, label_config=label_config)
    except Exception as e:
        context.logger.error("Error initializing model in webhook: " + str(e))
        return context.Response(
            body=json.dumps({"error": "Model initialization failed: " + str(e), "status": "error"}),
            headers={"Content-Type": "application/json"},
            status_code=500
        )

    try:
        result = webhook_model.fit(action, data)
    except Exception as e:
        context.logger.error("Error during model.fit in webhook: " + str(e))
        return context.Response(
            body=json.dumps({"error": str(e), "status": "error"}),
            headers={"Content-Type": "application/json"},
            status_code=500
        )

    response_data = {"result": result, "status": "ok"}
    return context.Response(
        body=json.dumps(response_data),
        headers={"Content-Type": "application/json"},
        status_code=201
    )

def handle_train(context, event):
    if event.body is None:
        context.logger.error("No training request body provided")
        return context.Response(
            body=json.dumps({"error": "No training request body provided"}),
            headers={"Content-Type": "application/json"},
            status_code=400
        )
    if isinstance(event.body, bytes):
        body_str = event.body.decode("utf-8")
    elif isinstance(event.body, dict):
        body_str = json.dumps(event.body)
    else:
        body_str = event.body
    try:
        data = json.loads(body_str)
    except Exception as e:
        context.logger.error("Failed to parse train request body: " + str(e))
        return context.Response(
            body=json.dumps({"error": "Invalid JSON in train request"}),
            headers={"Content-Type": "application/json"},
            status_code=400,
        )
    project = data.get("project")
    label_config = data.get("label_config")
    training_data = data.get("training_data")
    if training_data is None:
        context.logger.error("Training data is missing.")
        return context.Response(
            body=json.dumps({"error": "Training data is required"}),
            headers={"Content-Type": "application/json"},
            status_code=400,
        )
    try:
        new_model = Sam2Nuclio(project_id=project, label_config=label_config)
        train_result = new_model.train(training_data, **data.get("kwargs", {}))
    except Exception as e:
        context.logger.error("Error during training: " + str(e))
        return context.Response(
            body=json.dumps({"error": str(e)}),
            headers={"Content-Type": "application/json"},
            status_code=500,
        )
    return context.Response(
        body=json.dumps(train_result, ensure_ascii=False),
        headers={"Content-Type": "application/json"},
        status_code=200,
    )
