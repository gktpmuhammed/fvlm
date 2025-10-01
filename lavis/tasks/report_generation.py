from lavis.tasks.base_task import BaseTask
from lavis.common.registry import registry
from lavis.datasets.data_utils import prepare_sample

@registry.register_task("report_generation")
class ReportGenerationTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate=False, cuda_enabled=True):
        super().__init__()
        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate
        self.cuda_enabled = cuda_enabled

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        return cls(
            num_beams=run_cfg.num_beams,
            max_len=run_cfg.max_length,
            min_len=run_cfg.min_length,
            evaluate=run_cfg.evaluate,
            cuda_enabled=run_cfg.get('device', 'cuda') == 'cuda',
        )

    def build_model(self, cfg):
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config)
        return model

    def valid_step(self, model, samples):
        results = []
        samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled)
        
        # run inference
        captions = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )

        img_ids = samples["image_id"]
        for caption, img_id in zip(captions, img_ids):
            results.append({"caption": caption, "image_id": int(img_id)})

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        """
        Return validation metrics for checkpoint saving.
        Since we don't have ground truth for comparison, we use epoch number
        as a simple increasing metric (later epochs are considered better).
        """
        if val_result is None or len(val_result) == 0:
            return {"agg_metrics": 0.0}
        
        num_reports = len(val_result)
        
        # Use epoch as metric - later epochs are considered better
        # This ensures checkpoints are saved for each epoch
        agg_metrics = float(epoch) + 1.0  # +1 to avoid 0 for epoch 0
        
        return {
            "agg_metrics": agg_metrics,
            "num_reports": num_reports,
            "epoch": epoch,
        }
