from flask import Flask, render_template, request
from model import predict_segment, get_cluster_graph, get_statistics

app = Flask(__name__)

@app.route("/")
def home():

    graph = get_cluster_graph()
    stats = get_statistics()

    return render_template("index.html", graph=graph, stats=stats)

@app.route("/predict", methods=["POST"])
def predict():

    income = float(request.form["income"])
    score = float(request.form["score"])

    segment, explanation = predict_segment(income, score)

    graph = get_cluster_graph()

    return render_template(
        "result.html",
        segment=segment,
        explanation=explanation,
        graph=graph
    )

@app.route("/dataset")
def dataset():

    graph = get_cluster_graph()

    return render_template("dataset.html", graph=graph)

if __name__ == "__main__":
    app.run(debug=True)