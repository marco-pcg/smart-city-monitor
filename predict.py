"""Unified CLI for the Smart City Monitor models."""
import argparse
import json
import sys
from pathlib import Path

from inference import cnn_traffic, lstm_energy, gnn_fraud, transformer_reports, autoencoder_leaks


def cmd_cnn(args):
    print(json.dumps(cnn_traffic.predict(args.image), indent=2))


def cmd_lstm(args):
    with open(args.history) as f:
        values = json.load(f)
    print(json.dumps(lstm_energy.forecast(values), indent=2))


def cmd_gnn(args):
    with open(args.graph) as f:
        g = json.load(f)
    print(json.dumps(gnn_fraud.predict_nodes(g["features"], g["edges"]), indent=2))


def cmd_reports(args):
    if args.batch:
        with open(args.batch) as f:
            texts = json.load(f)
        print(json.dumps(transformer_reports.route_batch(texts), indent=2))
    else:
        print(json.dumps(transformer_reports.route(args.text), indent=2))


def cmd_leaks(args):
    with open(args.readings) as f:
        readings = json.load(f)
    print(json.dumps(autoencoder_leaks.detect(readings), indent=2))


def main():
    parser = argparse.ArgumentParser(description="Smart City Monitor — Unified Inference CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("traffic", help="Classify a traffic sign image")
    p.add_argument("image", help="Path to image file")
    p.set_defaults(func=cmd_cnn)

    p = sub.add_parser("energy", help="Forecast next-day energy demand")
    p.add_argument("history", help="Path to JSON file with 30 past values")
    p.set_defaults(func=cmd_lstm)

    p = sub.add_parser("fraud", help="Predict fraud labels for a graph")
    p.add_argument("graph", help="Path to JSON with 'features' and 'edges'")
    p.set_defaults(func=cmd_gnn)

    p = sub.add_parser("route", help="Route a citizen report")
    p.add_argument("--text", help="Single report text")
    p.add_argument("--batch", help="Path to JSON file with a list of texts")
    p.set_defaults(func=cmd_reports)

    p = sub.add_parser("leak", help="Detect water leak from sensor readings")
    p.add_argument("readings", help="Path to JSON file with sensor values")
    p.set_defaults(func=cmd_leaks)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
