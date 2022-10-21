class TrainConfig():

    def __init__(self, max_epochs: int, early_stopping_patience: int, save_dir: str,
                 start_time: str, batch_size: int = 8, save_top_k_checkpoints: int = 2,
                 no_resume: bool = False, version: str = None, sub_version: str = None) -> None:
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.save_dir = save_dir
        self.start_time = start_time
        self.batch_size = batch_size
        self.save_top_k_checkpoints = save_top_k_checkpoints
        self.no_resume = no_resume
        self.version = version
        self.sub_version = sub_version

        if self.sub_version is not None and self.version is None:
            raise ValueError('You may not define --sub-version without --version!')


class TrainResult():

    def __init__(self, interrupted: bool) -> None:
        self.interrupted = interrupted
