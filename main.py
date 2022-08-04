from model import create_optimizer, train_model, test_model, get_training_dataloader, get_testing_dataloader, Net, LabeledImageLoader, create_model, save, load
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
TRACKING_URI="http://localhost:5000"
mlflow.set_tracking_uri(TRACKING_URI)

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
    # TODO: Replace with mlflow.log_image
    with tempfile.TemporaryDirectory() as tmpdirname:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot()
        fname = tmpdirname + f"/{name}.png"
        plt.savefig(fname)
        plt.close()
        mlflow.log_artifact(fname, category)


def _train(name : str, data : Path, epochs : int):
    
    # Auto logging only works for pyTorch Lightning
    # Needs to derive model from pytorch_lightning.LightningModule, and use pytorch_lightning.Trainer()
    #mlflow.pytorch.autolog()
    try:
        expid = mlflow.create_experiment(name)
    except mlflow.exceptions.RestException:
        print(f"An experiment with the same name already exists! Choose another name ...")
        return

    with mlflow.start_run(run_name="FirstRun", experiment_id=expid, tags={'git_commit': "#FIXED", 'training_type': 'new'}) as run:
        print(f"Using mlflow run {run.info} ...")
        model = create_model()
        optimizer = create_optimizer(model)

        # Log list of manifest files
        mlflow.log_text(f"{data/'manifest.txt'}", "input_data.txt")
        # Log manifest file
        #mlflow.log_artifact(data / "manifest.txt", "input_data")
        trainloader = get_training_dataloader(data)
        testloader = get_testing_dataloader(data)

        train_model(model, optimizer, epochs, trainloader)
        cm = test_model(model, testloader)

        # Store results in mlflow
        stats = analyze_confusion_matrix(cm)
        log_confusion_matrix(stats['confusion_matrix'], LabeledImageLoader.labels, "confusion_matrix", "confusion_matrix")
        log_confusion_matrix(stats['normalized_confusion_matrix'], LabeledImageLoader.labels, "normalized_confusion_matrix", "confusion_matrix")
        del stats['confusion_matrix']
        del stats['normalized_confusion_matrix']

        mlflow.log_metrics(stats)
        #mlflow.pytorch.log_model(model, "model")
        save(model, optimizer, epochs, "model.pt")
        mlflow.log_artifact("model.pt", "model")

    # Register model
    model_uri = "runs:/{}/model".format(run.info.run_id)
    mv = mlflow.register_model(model_uri, "CIFAR10_pytorch")
    print("Name: {}".format(mv.name))
    print("Version: {}".format(mv.version))

def _finetune(run_id : str, data : Path, epochs : int = 2 ) :
    try:
        run_info = mlflow.get_run(run_id)
    except mlflow.exception.RestException:
        print(f"Run with id {run_id} not found!")
        return
    print(f"Fine tuning run {run_id}")

    with open(mlflow.artifacts.download_artifacts(artifact_path="input_data.txt", run_id=run_id)) as f:
        input_data = f.readlines()
    
    #model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    checkpoint = mlflow.artifacts.download_artifacts(artifact_path="model/model.pt", run_id=run_id)  
    model, optimizer, epochs = load(checkpoint)

    print(model)

    print(input_data)

@app.command()
def train(name : str, data : Path, epochs : int = 2):
   _train(name, data, epochs)

@app.command()
def show_runs(experiment : str):
    exp_id = mlflow.get_experiment_by_name(experiment)
    if exp_id:
        runs = mlflow.list_run_infos(experiment_id=exp_id.experiment_id)
        run_info = [(r.status, r.run_id) for r in runs]
        print(f"All Runs:\n{run_info}")
    else:
        print(f"No experiment found with the name: {experiment}")
        all_exp = [e.name for e in mlflow.list_experiments()]
        print(f"All found experiments:\n{all_exp}")


@app.command()
def fine_tune(run_id : str, data : Path, epochs : int = 2):
    _finetune(run_id, data, epochs)

def main():
    app()

if __name__ == "__main__":
    main()