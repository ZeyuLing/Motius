"""Image resizing and detection-result coordinates around a neural bundle."""
import torch


class PromptDetector:
    def __init__(self,bundle):
        from detectron2.data.transforms import ResizeShortestEdge
        self.bundle=bundle
        cfg=bundle.config
        self.resize=ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST,cfg.INPUT.MIN_SIZE_TEST],cfg.INPUT.MAX_SIZE_TEST)
        self.input_format=cfg.INPUT.FORMAT

    @torch.no_grad()
    def __call__(self,image):
        from detectron2.modeling.postprocessing import detector_postprocess
        if self.input_format=='RGB':image=image[:,:,::-1]
        height,width=image.shape[:2]
        resized=self.resize.get_transform(image).apply_image(image)
        tensor=torch.as_tensor(resized.astype('float32').transpose(2,0,1))
        inputs={'image':tensor,'height':height,'width':width}
        prediction=self.bundle([inputs])[0]
        return {'instances':detector_postprocess(prediction,height,width)}
