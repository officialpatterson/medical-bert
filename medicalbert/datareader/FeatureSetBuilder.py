# This class allows us to create a set of input features
# where each new input feature is added as another dimension
# For example
# We could add a value for the first element related to age
# and in the second element its health condition
# But for now, this allows us to break long pieces of text
# into shorter chunks

class FeatureSetBuilder:
    def __init__(self, label):
        self.features = []
        self.label = label

    def add(self, input_feature):
        self.features.append(input_feature.get_matrix())

    def resize(self, num_sections):
        # if the num sections isn't maxed we either need to pad out or cut down.
        if len(self.features) < num_sections:
            self.features.append(self.convert_section_to_feature([0], self.label))

        # Handle the case where we have too many sections - cut at the head
        if len(self.features) > num_sections:
            self.features = self.features[-num_sections:]

    #returns all features
    def get(self):
        return self.features

    def get_feature(self, feature_index):
        return self.features[feature_index]

    def get_label(self):
        return self.label