import os

from graph_similarity_platform import create_app


app = create_app()


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    port = int(os.environ.get("PORT", "5002"))
    app.run(host="127.0.0.1", port=port, debug=debug)
