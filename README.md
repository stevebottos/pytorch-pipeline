# What's in here?

* **references** folder is from here: https://github.com/pytorch/vision/tree/master/references. As you can see we've taken just 
  the detection portion of it. Some modifications have been made to detection/engine.py to allow for SSD training. We can keep this up 
  to date with Pytorch's references folder as they add new stuff. This is why I'm keeping it as its own separate folder. We're only using detection right now but the rest will be good to have on hand.
* **SSD** this is a blend of two repos https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD and https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. More specifically we've taken some elements of training from sgrvinod's repo and moved it into the references/detection/engine.py module, and re-worked some elements of the original nvidia SSD implementation to work with that. I've also redesigned things slightly so that the SSD will work with any input size rather than just the original SSD300 (hence the renaming)


