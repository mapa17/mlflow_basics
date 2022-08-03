from model import train_model, test_model, get_training_dataloader, get_testing_dataloader, Net, LabeledImageLoader
from pathlib import Path
from typing import Dict, List
import numpy
import typer
import tempfile

from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import mlflow

# Prior Start mlflow server locally with
# mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root mlflow/mlruns --host 0.0.0.0 --port 5000

app = typer.Typer()

def analyze_confusion_matrix(cm : numpy.ndarray) -> Dict[str, float]:
        ncm = cm / cm.sum()
        worst = ncm.diagonal().min()
        #worst_class = LabeledImageLoader.labels[ncm.diagonal().argmin()]
        worst_class = ncm.diagonal().argmin()
        median = numpy.median(ncm.diagonal())
        results = {
            'confusion_matrix': cm,
            'normalized_confusion_matrix': ncm,
            'worst_class': worst_class,
            'worst_class_score': worst,
            'median_score': median
        }
        return results

def log_confusion_matrix(cm : numpy.ndarray, labels : List[str], name : str, category : str) :
    with tempfile.TemporaryDirectory() as tmpdirname:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot()
        fname = tmpdirname + f"/{name}.png"
        plt.savefig(fname)
        plt.close()
        mlflow.log_artifact(fname, category)


def _train(name : str, data : Path, epochs : int):
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Auto logging only works for pyTorch Lightning
    # Needs to derive model from pytorch_lightning.LightningModule, and use pytorch_lightning.Trainer()
    #mlflow.pytorch.autolog()
    with mlflow.start_run(run_name=name, tags={'git_commit': "#FIXED", 'training_type': 'new'}) as run:
        print(f"Using mlflow run {run.info} ...")
        model = Net()

        trainloader = get_training_dataloader(data)
        testloader = get_testing_dataloader(data)

        train_model(model, epochs, trainloader)
        cm = test_model(model, testloader)

        # Store results in mlflow
        stats = analyze_confusion_matrix(cm)
        log_confusion_matrix(stats['confusion_matrix'], LabeledImageLoader.labels, "confusion_matrix", "confusion_matrix")
        log_confusion_matrix(stats['normalized_confusion_matrix'], LabeledImageLoader.labels, "normalized_confusion_matrix", "confusion_matrix")
        del stats['confusion_matrix']
        del stats['normalized_confusion_matrix']

        mlflow.log_metrics(stats)
        mlflow.pytorch.log_model(model, "model")


@app.command()
def train(name : str, data : Path, epochs : int = 2):
   _train(name, data, epochs)

@app.command()
def fine_tune(experiment : str):
    print(f"Fine tuning experiment {experiment}!")

def main():
    app()

if __name__ == "__main__":
    main()