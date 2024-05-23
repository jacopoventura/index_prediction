from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
import pickle


class PatternMatching:
    # initializer without train
    def __init__(self):
        self.ACCURACY_THRESHOLD_PATTERN_PREDICTION = .8
        self.PATTERN_MATCHING_ACCURACY = .9
        self.learned_patterns_dict = None
        self.train_data = None

    def set_minimum_prediction_accuracy(self, minimum_prediction_accuracy: float) -> None:
        """
        Set the minimum accuracy value for pattern prediction.
        :param minimum_prediction_accuracy: accuracy threshold
        :return: None
        """
        self.ACCURACY_THRESHOLD_PATTERN_PREDICTION = minimum_prediction_accuracy

    def set_minimum_pattern_matching_accuracy(self, minimum_matching_accuracy: float) -> None:
        """
        Set the minimum accuracy value for pattern matching.
        :param minimum_matching_accuracy: accuracy threshold
        :return: None
        """
        self.PATTERN_MATCHING_ACCURACY = minimum_matching_accuracy

    def set_train_data(self, train_data: list = None) -> None:
        """
        Set the training data.
        :param train_data: training data as list
        :return: None
        """
        self.train_data = train_data

    def set_learned_patterns(self, learned_patterns_dict: dict) -> None:
        """
        Set input dictionary as learned pattern dictionary.
        :param learned_patterns_dict: dictionary of learned patterns
        :return: None
        """
        self.learned_patterns_dict = learned_patterns_dict

    def export_learned_patterns(self, filename_without_extension: str) -> None:
        """
        Save learned pattern (dictionary format) into pickle file
        :param filename_without_extension: filename
        :return: None
        """
        with open(filename_without_extension + '.pkl', 'wb') as f:
            pickle.dump(self.learned_patterns_dict, f)

    def load_learned_patterns(self, filename_without_extension: str) -> None:
        """
        Load learned patterns from pickle file.
        :param filename_without_extension: filename
        :return: None
        """
        with open(filename_without_extension + '.pkl', 'rb') as f:
            self.learned_patterns_dict = pickle.load(f)

    def train(self, pattern_lengths_list: list) -> dict:
        """
        Train the model.
        :param pattern_lengths_list: list of pattern lengths
        :return: dictionary of the learned patterns.
        """
        len_train = len(self.train_data)
        learned_pattern_list = []
        learned_pattern_prediction_list = []
        learned_pattern_prediction_accuracy_list = []

        for step in pattern_lengths_list:
            pattern_list = []
            pattern_prediction_list = []
            pattern_prediction_accuracy_list = []
            for i in range(len_train - step):
                pattern_end_index = i + step

                # list of found patterns (these patterns shall be learned)
                sample = self.train_data[i:pattern_end_index]

                # check if sample has been already found
                if sample not in pattern_list:
                    sample_check = self.train_data[pattern_end_index + 1]
                    cumulative_score = 0
                    counter_valid_patterns = 0
                    # look ahead in the training dataset for the same pattern
                    for j in range(len_train - pattern_end_index - 1):
                        # check if the selected window matches the sample pattern (accuracy higher than threshold)
                        if accuracy_score(self.train_data[j:j + step], sample) >= self.PATTERN_MATCHING_ACCURACY:
                            # pattern found. Check if next day is the same as in the sample
                            if self.train_data[j + step + 1] == sample_check:
                                cumulative_score += 1
                            counter_valid_patterns += 1
                    if counter_valid_patterns > 0:  # here we can set a minimum amount of matched patterns
                        accuracy = cumulative_score / counter_valid_patterns
                        if accuracy >= self.ACCURACY_THRESHOLD_PATTERN_PREDICTION:
                            pattern_list.append(sample)
                            pattern_prediction_list.append(sample_check)
                            pattern_prediction_accuracy_list.append(accuracy)
            learned_pattern_list += pattern_list
            learned_pattern_prediction_list += pattern_prediction_list
            learned_pattern_prediction_accuracy_list += pattern_prediction_accuracy_list
        learned_patterns_df = {"pattern": learned_pattern_list,
                               "prediction": learned_pattern_prediction_list,
                               "accuracy": learned_pattern_prediction_accuracy_list}
        return learned_patterns_df

    def predict(self, past_data: list) -> tuple:
        """
        Predict given input data.
        :param past_data: input data to make the prediction
        :return: (prediction, probability)
        """

        length_past_data = len(past_data)

        highest_match_score = 0
        prediction = 1  # initialize with 1 since positive day has historically 54% probability
        probability_prediction = 0
        for pattern_idx in range(len(self.learned_patterns_dict["pattern"])):
            # select one learned pattern
            pattern = self.learned_patterns_dict["pattern"][pattern_idx]
            pattern_prediction = self.learned_patterns_dict["prediction"][pattern_idx]
            prediction_probability = self.learned_patterns_dict["accuracy"][pattern_idx]
            len_pattern = len(pattern)

            # access past data of the same length as the selected pattern and check pattern matching
            # past_data = test_data[(i - len_pattern):i]

            if length_past_data >= len_pattern:
                match_score = accuracy_score(past_data[(length_past_data - len_pattern):], pattern)
                if highest_match_score <= match_score:
                    if prediction_probability >= probability_prediction:
                        prediction = pattern_prediction
                        probability_prediction = prediction_probability
                        highest_match_score = match_score

        return prediction, probability_prediction

    def test(self, test_data: list):
        """
        Predict using the learned patterns.
        :param test_data: test dataset
        :return:
        """
        # get the maximum length among the learned patterns
        max_pattern_length = 0
        for pattern in self.learned_patterns_dict["pattern"]:
            pattern_length = len(pattern)
            if pattern_length > max_pattern_length:
                max_pattern_length = pattern_length

        length_test_data = len(test_data)
        prediction_list = [None] * (length_test_data - max_pattern_length)

        # I want to predict the selected element, therefore I need to start from max(pattern_length_list),
        # which corresponds to the (max(pattern_length_list) + 1)th element
        for i in range(max_pattern_length, length_test_data):

            # access past data and check pattern matching
            past_data = test_data[(i - max_pattern_length):i]
            prediction, _ = self.predict(past_data)
            prediction_list[i - max_pattern_length] = prediction

        accuracy = accuracy_score(test_data[max_pattern_length:], prediction_list)
        precision = precision_score(test_data[max_pattern_length:], prediction_list)
        tn, fp, fn, tp = confusion_matrix(test_data[max_pattern_length:], prediction_list).ravel()
        specificity = 0.
        if (tn + fp) > 0:
            specificity = tn / (tn + fp)

        print("Precision: ", precision, " specificity: ", specificity, " accuracy: ", accuracy)
