from torch import nn
import timm


def get_model(name, classes, training):
    if name == 'tinynet':
        return MultilabelTinyNet(classes, training)
    else:
        return MultilabelMobileOne(classes, training)


class MultilabelTinyNet(nn.Module):
    def __init__(self, classes, training):
        super(MultilabelTinyNet, self).__init__()
        self.tinynet = timm.create_model('tinynet_a', pretrained=training)
        self.classes = classes
        self.linear1 = nn.Linear(1000, 100)
        if isinstance(classes, dict) and len(classes) == 1:
            self.classifier = nn.Linear(100, len(classes[list(classes.keys())[0]]))
            nn.init.normal_(self.classifier.weight, 0, 0.01)
        else:
            self.hole = nn.Linear(100, 2)
            self.growth = nn.Linear(100, 3)
            nn.init.normal_(self.hole.weight, 0, 0.01)
            nn.init.normal_(self.growth.weight, 0, 0.01)
        self.relu = nn.ReLU()
        nn.init.normal_(self.linear1.weight, 0, 0.01)

    def forward(self, x):
        output = self.tinynet(x)
        if len(self.classes) == 1:
            return {list(self.classes.keys())[0]: self.classifier(self.relu(self.linear1(output)))}
        else:
            return {'holes': self.hole(self.relu(self.linear1(output))),
                    'growth': self.growth(self.relu(self.linear1(output)))}


class MultilabelMobileOne(nn.Module):
    def __init__(self, classes, training):
        super(MultilabelMobileOne, self).__init__()
        self.mobileOne = timm.create_model('mobileone_s2', pretrained=training)
        self.linear1 = nn.Linear(1000, 100)
        self.classes = classes
        if isinstance(classes, dict) and len(classes) == 1:
            self.classifier = nn.Linear(100, len(classes[list(classes.keys())[0]]))
            nn.init.normal_(self.classifier.weight, 0, 0.01)
        else:
            self.hole = nn.Linear(100, 2)
            self.growth = nn.Linear(100, 3)
            nn.init.normal_(self.hole.weight, 0, 0.01)
            nn.init.normal_(self.growth.weight, 0, 0.01)
        self.relu = nn.ReLU()
        nn.init.normal_(self.linear1.weight, 0, 0.01)

    def forward(self, x):
        output = self.mobileOne(x)

        if len(self.classes) == 1:
            return {list(self.classes.keys())[0]: self.classifier(self.relu(self.linear1(output)))}
        else:
            return {'holes': self.hole(self.relu(self.linear1(output))),
                    'growth': self.growth(self.relu(self.linear1(output)))}