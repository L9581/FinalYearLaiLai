**This project aims to develop an autonomous task execution system that integrates multimodal human–agent interaction, vision-language models (VLMs), PDDL-based planning, and executable skill invocation. Although recent VLMs can understand natural-language instructions and visual scenes, converting ambiguous, high-level user requests into reliable action sequences remains challenging.**



**The proposed system will accept textual, spoken, or visual instructions and use a VLM to identify task-relevant objects, attributes, spatial relations, and environmental states. A dialogue module will interpret user intentions and request clarification when goals or conditions are incomplete. The agent will then transform the perceived scene and user goal into a structured symbolic state and construct a PDDL problem using predefined predicates, action preconditions, and effects. A PDDL planner will generate a high-level action sequence, while a skill manager will map each symbolic action to an executable capability, such as object detection, navigation, grasping, placing, or software-tool invocation.**



**Inspired by the ReAct architecture, execution will follow a reasoning–action–observation loop. The agent will monitor action outcomes, update its world state, diagnose failures, and replan when necessary. The system will be implemented in Python, connected to a VLM/LLM and a PDDL planner such as Fast Downward, and evaluated in simulated or simplified robotic scenarios. Expected outcomes include a working prototype, reusable planning and skill models, and experimental evaluation of understanding accuracy, planning success, task completion, and failure recovery.**

