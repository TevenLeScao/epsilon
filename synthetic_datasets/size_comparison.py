import argparse

from utils import fill_in_template, create_text_jsonl, integrate_within_existing_data


class SizeComparisonDataset:

    def __init__(self):

        self.same_size_objects = ["penguin", "pangolin", "raccoon", "snake", "fox", "chair", "suitcase", "stove", "radiator"]
        self.abstract_objects = ["thought", "concept", "dream", "number", "joke", "defeat", "relationship", "song", "color", "scent"]
        self.all_objects = self.same_size_objects + self.abstract_objects

        self.singular_templates = ["a $1 is bigger than a $2", "a $2 is smaller than a $1", "a $1 is larger than a $2",
                              "a $2 is shorter than a $1"]
        self.plural_templates = ["$1s are bigger than $2s", "$2s are smaller than $1s", "$1s are larger then $2s",
                            "$2s are shorter than $1s"]
        self.all_templates = self.singular_templates + self.plural_templates

        self.realizations = [fill_in_template(template, object1, object2) for object1 in self.all_objects for object2 in self.all_objects for template in self.all_templates]
        self.n_relations = len(self.realizations)
        self.n_asym_relations = len(self.all_templates) * len(self.all_objects) * (len(self.all_objects) - 1)
        self.n_objects = len(self.all_objects)
        self.n_templates = len(self.all_templates)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.realizations[item]
        if isinstance(item, tuple):
            if len(item) != 3:
                raise ValueError(f"you should pass a 3-tuple identifier instead of {item} of length {len(item)}")
            else:
                return fill_in_template(self.all_templates[item[0]], self.all_objects[item[1]],
                                        self.all_objects[item[2]])

    def get_related_sentences(self, identifier):
        if len(identifier) != 3:
            raise ValueError(
                f"you should pass a 3-tuple identifier instead of {identifier} of length {len(identifier)}")
        tp, obj1, obj2 = identifier
        main_sentence = self[identifier]
        support_sentences = [self[extra_tp, obj1, obj2] for extra_tp in range(len(self.all_templates)) if
                                extra_tp != tp]
        if obj1 == obj2:
            counter_sentences = []
        else:
            # putting the exact opposite sentence first
            counter_sentences = [self[tp, obj2, obj1]] + [self[extra_tp, obj2, obj1] for extra_tp in
                                                          range(len(self.all_templates)) if extra_tp != tp]
        return main_sentence, support_sentences, counter_sentences


sorted_size_objects = ["electron", "proton", "atom", "molecule", "cell", "worm", "bird", "horse", "elephant", "whale",
                       "island", "continent", "moon", "planet", "star"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_filepath", type=str, required=True, help="where to export the train jsonl")
    parser.add_argument("--valid_support_filepath", type=str, required=True, help="where to export the train jsonl")
    parser.add_argument("--valid_counter_filepath", type=str, required=True, help="where to export the train jsonl")
    parser.add_argument("--tp", type=int, default=0, help="choose a template from the dataset")
    parser.add_argument("--obj1", type=int, default=0, help="choose an object from the dataset")
    parser.add_argument("--obj2", type=int, default=1, help="choose a second object from the dataset")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--single_index", type=int, default=None, help="one-integer definition of seed, template, and objects")
    parser.add_argument("--existing_data_path", default=None, type=str)
    parser.add_argument("--min_size", default=50000, type=int)
    args = parser.parse_args()

    dataset = SizeComparisonDataset()
    print(dataset.n_asym_relations, dataset.n_templates, dataset.n_objects)
    if args.single_index is None:
        tp, obj1, obj2, seed = args.tp, args.obj1, args.obj2, args.seed
    else:
        remainder = args.single_index
        print(remainder)
        seed = remainder // dataset.n_asym_relations
        remainder = remainder % dataset.n_asym_relations
        print(remainder)
        tp = remainder % dataset.n_templates
        remainder = remainder // dataset.n_templates
        print(remainder)
        obj1 = remainder % dataset.n_objects
        remainder = remainder // dataset.n_objects
        print(remainder)
        obj2 = [i for i in range(dataset.n_objects) if i != obj1][remainder]

    print(f"seed: {seed} | template: {tp} | obj1: {obj1} | obj2: {obj2}")

    main_sentence, support_sentences, counter_sentences = dataset.get_related_sentences((tp, obj1, obj2))
    if args.existing_data_path is None:
        create_text_jsonl(args.train_filepath, main_sentence, min_size=args.min_size)
    else:
        combined_data = integrate_within_existing_data(args.train_filepath, main_sentence, existing_data_path=args.existing_data_path, seed=seed)
    create_text_jsonl(args.valid_support_filepath, support_sentences, min_size=args.min_size)
    create_text_jsonl(args.valid_counter_filepath, counter_sentences, min_size=args.min_size)
