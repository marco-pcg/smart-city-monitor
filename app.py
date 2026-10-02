"""
Gradio demo for the Smart City Monitor.
Each model gets its own tab.

Run: python app.py
"""
import json
import gradio as gr

from inference import (
    cnn_traffic,
    lstm_energy,
    gnn_fraud,
    transformer_reports,
    autoencoder_leaks,
)

# ---------- Module 01: CNN ----------
def cnn_fn(image):
    if image is None:
        return {"error": "No image provided"}
    result = cnn_traffic.predict(image)
    return {item["label"]: item["confidence"] for item in result["top_k"]}


# ---------- Module 02: LSTM ----------
def lstm_fn(history_text):
    try:
        values = [float(x.strip()) for x in history_text.replace("\n", ",").split(",") if x.strip()]
        result = lstm_energy.forecast(values)
        return f"Predicted next-day consumption: {result['forecast_kwh']:.2f} kWh"
    except Exception as e:
        return f"❌ Error: {e}"


# ---------- Module 03: GNN ----------
def gnn_fn(features_text, edges_text):
    try:
        features = json.loads(features_text)
        edges = json.loads(edges_text)
        result = gnn_fraud.predict_nodes(features, edges)
        lines = [f"Node {p['node']}: {p['label']} (fraud prob: {p['fraud_probability']:.2%})"
                 for p in result["predictions"]]
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Error: {e}"


# ---------- Module 04: Transformer ----------
def transformer_fn(text):
    result = transformer_reports.route(text)
    return result["department"], result["all_scores"]


# ---------- Module 05: Autoencoder ----------
def leak_fn(readings_text):
    try:
        readings = [float(x.strip()) for x in readings_text.split(",") if x.strip()]
        result = autoencoder_leaks.detect(readings)
        emoji = {"normal": "🟢", "moderate": "🟡", "high": "🔴"}[result["severity"]]
        return (
            f"{emoji} Severity: {result['severity'].upper()}\n"
            f"Anomaly score: {result['anomaly_score']:.6f}\n"
            f"Threshold:     {result['threshold']:.6f}\n"
            f"Is leak:       {result['is_leak']}"
        )
    except Exception as e:
        return f"❌ Error: {e}"


# ---------- Build UI ----------
with gr.Blocks(title="Smart City Monitor") as demo:
    gr.Markdown("# 🏙️ Smart City Monitor")
    gr.Markdown("Five neural network architectures in one demo.")

    with gr.Tab("🚦 Traffic Sign (CNN)"):
        with gr.Row():
            img_in = gr.Image(type="filepath", label="Upload a traffic sign")
            lbl_out = gr.Label(num_top_classes=3, label="Prediction")
        img_in.change(cnn_fn, inputs=img_in, outputs=lbl_out)

    with gr.Tab("⚡ Energy Forecast (LSTM)"):
        hist_in = gr.Textbox(
            label="Last N daily kWh values (comma-separated)",
            value="1200,1150,1300,1250,1400,1350,1280,1220,1190,1310,1420,1380,1260,1240,1210,1330,1450,1400,1290,1270,1230,1180,1300,1440,1390,1270,1250,1220,1200,1340",
            lines=3,
        )
        hist_out = gr.Textbox(label="Forecast")
        gr.Button("Predict").click(lstm_fn, inputs=hist_in, outputs=hist_out)

    with gr.Tab("🕸️ Fraud Detection (GNN)"):
        with gr.Row():
            feat_in = gr.Code(
                label="Node features (JSON)",
                value='[[1.0,0.5],[0.5,1.0],[0.1,0.1],[5.0,5.0]]',
                language="json",
            )
            edge_in = gr.Code(
                label="Edge index (JSON)",
                value='[[0,1,2,3],[1,2,3,0]]',
                language="json",
            )
        gnn_out = gr.Textbox(label="Predictions", lines=6)
        gr.Button("Detect").click(gnn_fn, inputs=[feat_in, edge_in], outputs=gnn_out)

    with gr.Tab("📨 Citizen Reports (Transformer)"):
        text_in = gr.Textbox(
            label="Citizen report",
            value="There is a broken pipe leaking water on Elm Street.",
            lines=2,
        )
        dept_out = gr.Textbox(label="Routed to")
        scores_out = gr.Label(label="All departments")
        gr.Button("Route").click(transformer_fn, inputs=text_in, outputs=[dept_out, scores_out])

    with gr.Tab("💧 Leak Detection (Autoencoder)"):
        read_in = gr.Textbox(
            label="Sensor readings (comma-separated)",
            value="5.0,5.0,0.1,0.1,5.0,5.0,0.1,0.1,5.0,5.0",
            lines=2,
        )
        leak_out = gr.Textbox(label="Result", lines=4)
        gr.Button("Detect").click(leak_fn, inputs=read_in, outputs=leak_out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
