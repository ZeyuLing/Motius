from motius.evaluation.babel import (
    BabelActionCatalog,
    build_action_group,
    positive_group_id,
)


def test_official_action_categories_merge_caption_variants_but_keep_modifiers():
    annotations = {
        "1": {
            "seq_ann": {"labels": []},
            "frame_ann": {
                "labels": [
                    {
                        "raw_label": "walking",
                        "proc_label": "walk",
                        "act_cat": ["walk"],
                    },
                    {
                        "raw_label": "walking forward",
                        "proc_label": "walk forward",
                        "act_cat": ["walk"],
                    },
                    {
                        "raw_label": "walking back",
                        "proc_label": "walk back",
                        "act_cat": ["walk", "backwards movement"],
                    },
                ]
            },
        }
    }
    catalog = BabelActionCatalog(annotations)
    walk = catalog.resolve("walking", babel_id="1", processed=False)
    forward = catalog.resolve("walking forward", babel_id="1", processed=False)
    backward = catalog.resolve("walking back", babel_id="1", processed=False)
    walk_id, _ = build_action_group([walk.categories], ["walking"])
    forward_id, _ = build_action_group([forward.categories], ["walking forward"])
    backward_id, _ = build_action_group([backward.categories], ["walking back"])
    assert walk_id == forward_id
    assert walk_id != backward_id


def test_positive_group_id_falls_back_to_exact_caption():
    first = positive_group_id({"caption": "A person walks."})
    second = positive_group_id({"caption": "a person walks"})
    different = positive_group_id({"caption": "A person walks back."})
    assert first == second
    assert first != different
