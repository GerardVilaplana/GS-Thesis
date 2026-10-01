from __future__ import annotations


HANDAL_OBJECT_NAMES = {
    "adjustable_wrenches": "adjustable wrench",
    "combinational_wrenches": "combinational wrench",
    "fixed_joint_pliers": "fixed joint pliers",
    "hammers": "hammer",
    "ladles": "ladle",
    "locking_pliers": "locking pliers",
    "measuring_cups": "measuring cup",
    "mugs": "mug",
    "pots_pans": "pot or pan",
    "power_drills": "power drill",
    "ratchets": "ratchet",
    "screwdrivers": "screwdriver",
    "slip_joint_pliers": "slip joint pliers",
    "spatulas": "spatula",
    "strainers": "strainer",
    "utensils": "utensil",
    "whisks": "whisk",
}


def normalize_object_name(name: str) -> str:
    return HANDAL_OBJECT_NAMES.get(name, name.replace("_", " ")).strip()


def official_csv_object_name(object_name: str) -> str | None:
    text = object_name.lower().replace("_", " ")
    if "mug" in text:
        return "mug"
    if "knife" in text:
        return "knife"
    return None


def make_fallback_grasp_prompt(object_name: str) -> tuple[str, str]:
    question = f"Identify the key points on the {object_name} that ensure a successful grasping experience."
    answer = f"The handle or graspable region of the {object_name} provides a secure place for grasping<Aff>."
    return question, answer


def make_paper_grasp_prompt(category_or_object: str, question_table=None) -> tuple[str, str, str, str]:
    object_name = normalize_object_name(category_or_object)
    official_obj = official_csv_object_name(object_name)
    if official_obj is not None and question_table is not None:
        row = question_table[(question_table["Object"] == official_obj) & (question_table["Affordance"] == "grasp")]
        if len(row) > 0:
            question = str(row["Question0"].values[0])
            answer = str(row["Answer2"].values[0])
            return question, answer, "official_csv_question0_answer2", official_obj
    question, answer = make_fallback_grasp_prompt(object_name)
    return question, answer, "paper_style_fallback_full_answer", object_name
