import torchvision.transforms as transforms

class BarlowTwinsValTransform:
    def __init__(self, input_height=224, normalize=None):
        self.input_height = input_height
        self.normalize = normalize

        if normalize is None:
            self.final_transform = transforms.ToTensor()
        else:
            self.final_transform = transforms.Compose([transforms.ToTensor(), normalize])

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.input_height),
                transforms.CenterCrop(self.input_height),
                self.final_transform,
            ]
        )

        # self.finetune_transform = transforms.ToTensor()

    def __call__(self, sample):
        return self.transform(sample)