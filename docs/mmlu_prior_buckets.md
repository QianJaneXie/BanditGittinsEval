# MMLU Prior Buckets

For MMLU, we keep the default vague prior `N(0.5, 0.04)` as the uninformative baseline. As an informative alternative, we use a coarse three-bucket prior rather than fitting a separate prior to each subject. This reflects approximate prior knowledge about subject difficulty while avoiding overly specific, decimal-valued tuning.


| MMLU bucket                              | Prior           | Interpretation                                                                   |
| ---------------------------------------- | --------------- | -------------------------------------------------------------------------------- |
| Low expected accuracy / hard subjects    | `N(0.4, 0.02)`  | Math, physics, formal logic, difficult STEM, and difficult professional subjects |
| Medium expected accuracy                 | `N(0.6, 0.02)`  | Mixed science, medicine, economics, philosophy, and general-knowledge subjects   |
| High expected accuracy / easier subjects | `N(0.75, 0.02)` | Many humanities, social science, policy, history, and management-style subjects  |


The bucket assignment uses simple coarse thresholds based on estimated row means:

- Low bucket: row mean `< 0.5`
- Medium bucket: row mean `0.5-0.7`
- High bucket: row mean `> 0.7`

## Low Bucket: `N(0.4, 0.02)`

- `abstract_algebra`
- `high_school_mathematics`
- `moral_scenarios`
- `college_mathematics`
- `college_physics`
- `high_school_physics`
- `global_facts`
- `formal_logic`
- `elementary_mathematics`
- `college_chemistry`
- `econometrics`
- `professional_law`
- `professional_accounting`
- `machine_learning`
- `high_school_chemistry`
- `high_school_statistics`
- `college_computer_science`
- `virology`

## Medium Bucket: `N(0.6, 0.02)`

- `conceptual_physics`
- `college_medicine`
- `electrical_engineering`
- `anatomy`
- `high_school_macroeconomics`
- `professional_psychology`
- `professional_medicine`
- `business_ethics`
- `high_school_computer_science`
- `high_school_microeconomics`
- `clinical_knowledge`
- `astronomy`
- `public_relations`
- `philosophy`
- `human_aging`
- `college_biology`
- `medical_genetics`
- `moral_disputes`
- `high_school_european_history`
- `nutrition`
- `prehistory`
- `security_studies`

## High Bucket: `N(0.75, 0.02)`

- `jurisprudence`
- `high_school_biology`
- `logical_fallacies`
- `human_sexuality`
- `computer_security`
- `management`
- `high_school_geography`
- `international_law`
- `world_religions`
- `high_school_us_history`
- `miscellaneous`
- `high_school_world_history`
- `high_school_psychology`
- `sociology`
- `high_school_government_and_politics`
- `us_foreign_policy`
- `marketing`

