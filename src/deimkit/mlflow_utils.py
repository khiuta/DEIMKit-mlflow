import mlflow

class MLflowWriter:
    """Duck-typed stand-in for SummaryWriter — only implements what det_engine.py actually calls."""
    def add_scalar(tag, scalar_value, global_step=None, *args, **kwargs):
        mlflow.log_metric(tag, float(scalar_value), step=global_step)

    def add_figure(self, tag, figure, global_step=None, *args, **kwargs):
        mlflow.log_figure(figure, f"{tag.replace('/', '_')}_step{global_step}.png")

    def close(self):
        pass

    def flush(self):
        pass
