import abc
import os
import glob
import random
import torch
import re

from typing import List, NamedTuple, Type
from libero.libero import get_libero_path
from libero.libero.benchmark.libero_suite_task_map import libero_task_map
import libero.libero.envs.bddl_utils as BDDLUtils

BENCHMARK_MAPPING = {}


def register_benchmark(target_class):
    """We design the mapping to be case-INsensitive."""
    BENCHMARK_MAPPING[target_class.__name__.lower()] = target_class


def get_benchmark_dict(help=False):
    if help:
        print("Available benchmarks:")
        for benchmark_name in BENCHMARK_MAPPING.keys():
            print(f"\t{benchmark_name}")
    return BENCHMARK_MAPPING


def get_benchmark(benchmark_name):
    return BENCHMARK_MAPPING[benchmark_name.lower()]


def print_benchmark():
    print(BENCHMARK_MAPPING)


class Task(NamedTuple):
    name: str
    language: str
    problem: str
    problem_folder: str
    bddl_file: str
    init_states_file: str


def grab_language_from_filename(suite_name, x):
    if "_language_" not in x:
        if x[0].isupper():  # LIBERO-100
            if "SCENE10" in x:
                language = " ".join(x[x.find("SCENE") + 8 :].split("_"))
            else:
                language = " ".join(x[x.find("SCENE") + 7 :].split("_"))
        else:
            language = " ".join(x.split("_"))
        en = language.find(".bddl")
        return language[:en]
    else:
        if "_view_" in x:
            bddl_file_path = os.path.join(
                get_libero_path("bddl_files"),
                suite_name,
                x.split("_view_")[0]+'.bddl',
            )
        else:
            bddl_file_path = os.path.join(
                get_libero_path("bddl_files"),
                suite_name,
                x,
            )
        # print("bddl_file_path:", bddl_file_path)
        problem_info = BDDLUtils.get_problem_info(bddl_file_path)
        return problem_info["language_instruction"]


libero_suites = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_10",
]

LIBERO_PRO_BASE_TASK_MAP = {
    "libero_spatial": [
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    ],
    "libero_object": [
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "pick_up_the_cream_cheese_and_place_it_in_the_basket",
        "pick_up_the_salad_dressing_and_place_it_in_the_basket",
        "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
        "pick_up_the_ketchup_and_place_it_in_the_basket",
        "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
        "pick_up_the_butter_and_place_it_in_the_basket",
        "pick_up_the_milk_and_place_it_in_the_basket",
        "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
        "pick_up_the_orange_juice_and_place_it_in_the_basket",
    ],
    "libero_goal": [
        "open_the_middle_drawer_of_the_cabinet",
        "put_the_bowl_on_the_stove",
        "put_the_wine_bottle_on_top_of_the_cabinet",
        "open_the_top_drawer_and_put_the_bowl_inside",
        "put_the_bowl_on_top_of_the_cabinet",
        "push_the_plate_to_the_front_of_the_stove",
        "put_the_cream_cheese_in_the_bowl",
        "turn_on_the_stove",
        "put_the_bowl_on_the_plate",
        "put_the_wine_bottle_on_the_rack",
    ],
    "libero_10": [
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    ],
}

LIBERO_PRO_PERTURBATIONS = ["lan", "object", "swap", "task", "env"]
LIBERO_PRO_BASE_SUITES = ["libero_goal", "libero_spatial", "libero_10", "libero_object"]
LIBERO_PRO_SUITES = [
    f"{base_suite}_{perturbation}"
    for perturbation in LIBERO_PRO_PERTURBATIONS
    for base_suite in LIBERO_PRO_BASE_SUITES
]

for libero_pro_suite in LIBERO_PRO_SUITES:
    base_suite = libero_pro_suite.rsplit("_", 1)[0]
    if base_suite in LIBERO_PRO_BASE_TASK_MAP and libero_pro_suite not in libero_task_map:
        libero_task_map[libero_pro_suite] = LIBERO_PRO_BASE_TASK_MAP[base_suite]
        libero_suites.append(libero_pro_suite)

task_maps = {}
max_len = 0
for libero_suite in libero_suites:
    task_maps[libero_suite] = {}

    for task in libero_task_map[libero_suite]:
        language = grab_language_from_filename(libero_suite, task + ".bddl")
        task_maps[libero_suite][task] = Task(
            name=task,
            language=language,
            problem="Libero",
            problem_folder=libero_suite,
            bddl_file=f"{task}.bddl",
            init_states_file=f"{task}.pruned_init",
        )

        # print(language, "\n", f"{task}.bddl", "\n")
        # print("")

suite_order = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]
task_num = [2402, 2518, 2591, 2519, 90]
task_order_dict = dict()

for idx in range(5):
    task_orders = [list(range(0,task_num[idx]))]
    for _ in range(19):
        order = list(range(0,task_num[idx]))
        random.shuffle(order)
        task_orders.append(order)
    task_order_dict[suite_order[idx]] = task_orders

class Benchmark(abc.ABC):
    """A Benchmark."""

    def __init__(self, task_order_index=0):
        self.task_embs = None
        self.task_order_index = task_order_index

    def _make_benchmark(self):
        tasks = list(task_maps[self.name].values())
        if self.name in task_order_dict:
            print(f"[info] using task orders {task_order_dict[self.name][self.task_order_index]}")
            self.tasks = [tasks[i] for i in task_order_dict[self.name][self.task_order_index]]
        else:
            print(f"[info] using default task order for {self.name}")
            self.tasks = tasks
        self.n_tasks = len(self.tasks)

    def get_num_tasks(self):
        return self.n_tasks

    def get_task_names(self):
        return [task.name for task in self.tasks]

    def get_task_problems(self):
        return [task.problem for task in self.tasks]

    def get_task_bddl_files(self):
        return [task.bddl_file for task in self.tasks]

    def get_task_bddl_file_path(self, i):
        bddl_file_path = os.path.join(
            get_libero_path("bddl_files"),
            self.tasks[i].problem_folder,
            self.tasks[i].bddl_file,
        )
        return bddl_file_path

    def get_task_demonstration(self, i):
        assert (
            0 <= i and i < self.n_tasks
        ), f"[error] task number {i} is outer of range {self.n_tasks}"
        # this path is relative to the datasets folder
        demo_path = f"{self.tasks[i].problem_folder}/{self.tasks[i].name}_demo.hdf5"
        return demo_path

    def get_task(self, i):
        return self.tasks[i]

    def get_task_emb(self, i):
        return self.task_embs[i]

    def get_task_init_states_ori(self, i):
        if "_table_" in self.tasks[i].init_states_file:
            init_states_path = os.path.join(
                get_libero_path("init_states"),
                self.tasks[i].problem_folder,
                self.tasks[i].init_states_file.split("_table_")[0] + "." + self.tasks[i].init_states_file.split(".")[-1],
            )
        elif "_tb_" in self.tasks[i].init_states_file:
            init_states_path = os.path.join(
                get_libero_path("init_states"),
                self.tasks[i].problem_folder,
                self.tasks[i].init_states_file.split("_tb_")[0] + "." + self.tasks[i].init_states_file.split(".")[-1],
            )
        elif "_view_" in self.tasks[i].init_states_file:
            init_states_path = os.path.join(
                get_libero_path("init_states"),
                self.tasks[i].problem_folder,
                self.tasks[i].init_states_file.split("_view_")[0] + "." + self.tasks[i].init_states_file.split(".")[-1],
            )
        else:
            init_states_path = os.path.join(
                get_libero_path("init_states"),
                self.tasks[i].problem_folder,
                self.tasks[i].init_states_file,
            )

        init_states = torch.load(init_states_path)
        return init_states
    
    def get_task_init_states(self, i):
        # print("======", re.sub(r'_table_\d+$', '', self.tasks[i].init_states_file))
        # print("====init_states_path=====", self.tasks[i].init_states_file)
        init_states_path = os.path.join(
            get_libero_path("init_states"),
            self.tasks[i].problem_folder,
            self.tasks[i].init_states_file,
        )
        if "_language_" in self.tasks[i].init_states_file:
            init_states_path = os.path.join(
                get_libero_path("init_states"),
                self.tasks[i].problem_folder,
                self.tasks[i].init_states_file.split("_language_")[0] + "." + self.tasks[i].init_states_file.split(".")[-1],
            )
        else:
            if "_view_" in self.tasks[i].init_states_file:
                init_states_path = os.path.join(
                    get_libero_path("init_states"),
                    self.tasks[i].problem_folder,
                    self.tasks[i].init_states_file.split("_view_")[0] + "." + self.tasks[i].init_states_file.split(".")[-1],
                )
            else:
                if "_table_" in self.tasks[i].init_states_file:
                    init_states_path = os.path.join(
                        get_libero_path("init_states"),
                        self.tasks[i].problem_folder,
                        re.sub(r'_table_\d+', '', self.tasks[i].init_states_file),
                    )
                if "_tb_" in self.tasks[i].init_states_file:
                    init_states_path = os.path.join(
                        get_libero_path("init_states"),
                        self.tasks[i].problem_folder,
                        re.sub(r'_tb_\d+', '', self.tasks[i].init_states_file),
                    )
                
                if "_light_" in self.tasks[i].init_states_file:
                    init_states_path = os.path.join(
                        get_libero_path("init_states"),
                        self.tasks[i].problem_folder,
                        self.tasks[i].init_states_file.split("_light_")[0] + "." + self.tasks[i].init_states_file.split(".")[-1],
                    )
                
                if "_add_" in self.tasks[i].init_states_file or "_level" in self.tasks[i].init_states_file:
                    init_states_path = os.path.join(
                        get_libero_path("init_states"),
                        "libero_newobj",
                        self.tasks[i].problem_folder,
                        self.tasks[i].init_states_file,
                    )
        # else:
        #     init_states_path = os.path.join(
        #         get_libero_path("init_states"),
        #         self.tasks[i].problem_folder,
        #         self.tasks[i].init_states_file,
        #     )
        
        # print("====init_states_path=====", init_states_path)

        init_states = torch.load(init_states_path, weights_only = False)
        if "_add_" in self.tasks[i].init_states_file or "_level" in self.tasks[i].init_states_file:
            init_states = init_states.reshape(1, -1)
        return init_states

    def set_task_embs(self, task_embs):
        self.task_embs = task_embs


@register_benchmark
class LIBERO_SPATIAL(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial"
        self._make_benchmark()


@register_benchmark
class LIBERO_OBJECT(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object"
        self._make_benchmark()


@register_benchmark
class LIBERO_GOAL(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal"
        self._make_benchmark()


@register_benchmark
class LIBERO_90(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        assert (
            task_order_index == 0
        ), "[error] currently only support task order for 10-task suites"
        self.name = "libero_90"
        self._make_benchmark()


@register_benchmark
class LIBERO_10(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10"
        self._make_benchmark()


@register_benchmark
class LIBERO_100(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_100"
        self._make_benchmark()

@register_benchmark
class LIBERO_MIX(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_mix"
        self._make_benchmark()


def _register_libero_pro_benchmark_class(suite_name):
    class_name = suite_name.upper()

    def __init__(self, task_order_index=0):
        Benchmark.__init__(self, task_order_index=task_order_index)
        self.name = suite_name
        self._make_benchmark()

    benchmark_cls = type(class_name, (Benchmark,), {"__init__": __init__})
    register_benchmark(benchmark_cls)
    globals()[class_name] = benchmark_cls


for _libero_pro_suite in LIBERO_PRO_SUITES:
    _register_libero_pro_benchmark_class(_libero_pro_suite)
