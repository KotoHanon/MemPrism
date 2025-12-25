from textwrap import dedent

WORKING_SLOT_EXPERIEMENT_FILTER_USER_PROMPT = dedent("""
You guard ResearchAgent's long-term memory entrance. Decide if this WorkingSlot deserves promotion into FAISS storage.

Assess four dimensions:
1. Novelty – is this meaningfully new compared to typical research agent discoveries?
2. Utility – can future tasks reuse the insight, metric, procedure, or decision?
3. Stability – will the information stay valid for multiple iterations (i.e., not a transient log)?
4. Evidence – do attachments, metrics, or tags provide concrete support?

Return `yes` only when at least two dimensions are clearly satisfied or the slot closes a critical loop (e.g., root-causing a failure, finishing a checklist item). Otherwise return `no`.

STRICT OUTPUT: respond with a single lowercase word: `yes` or `no`. Do not explain.

<slot-dump>
{slot_dump}
</slot-dump>
""")

WORKING_SLOT_QA_FILTER_USER_PROMPT = dedent("""
You guard QA-Agent's long-term memory entrance. Decide if this WorkingSlot deserves promotion into FAISS storage.

Your goal is to keep **reusable, high-level memories** that can help answer future questions in the same domain (e.g., HotpotQA-style multi-hop QA), not just low-level logs.

Evaluate the slot along three dimensions:

1. Reusable knowledge or pattern – Does this slot contain a fact, relation, strategy, or failure pattern that could be helpful for similar questions in the future?
   - Includes:
     - Entity attributes (dates, locations, roles, definitions).
     - Relations between entities (A part_of B, A located_in B, A spouse_of B, A causes B).
     - Short multi-hop bridges that connect entities or events.
     - Generalizable QA strategies or common failure modes.

2. Abstraction level – Is the content more than raw log text?
   - Prefer:
     - Summaries of reasoning steps, heuristics, or patterns.
     - Condensed statements that compress multiple observations.
   - Avoid:
     - Pure boilerplate (“now I will think step by step”, “searching Wikipedia…”).
     - Very low-level noise (token indices, random IDs, partial URLs) with no clear use.

3. Reliability and stability – Is the information well-supported by the context and relatively stable over time?
   - Prefer:
     - Facts or patterns that are explicitly stated or clearly implied.
   - Avoid:
     - Pure guesses, opinions, or very ephemeral states.

Decision rule:
- Default is slightly conservative, but you SHOULD store any slot that carries at least one **non-trivial, reusable** fact, relation, or strategy.
- Return `yes` if:
  - The slot contains **some reusable knowledge or pattern** (dimension 1), AND
  - It is not dominated by pure logging/noise, AND
  - It is at least moderately reliable (dimension 3) OR shows some abstraction (dimension 2).
- Return `no` if the slot is:
  - Mostly boilerplate/logging,
  - Extremely local and unlikely to be reused,
  - Or dominated by noise/IDs without a clear semantic core.

When you are uncertain BUT the slot contains at least one meaningful, reusable fact or strategy, prefer answering `yes` rather than `no`.

STRICT OUTPUT: respond with a single lowercase word: `yes` or `no`. Do not explain.

<slot-dump>
{slot_dump}
</slot-dump>
""")

WORKING_SLOT_FC_FILTER_USER_PROMPT = dedent("""
You guard a Function-Calling agent's long-term memory entrance. Decide if this candidate memory deserves promotion into FAISS storage.

Your goal is to keep **reusable, high-level function-calling knowledge** that improves future tool-use (schema grounding, argument filling, error recovery, tool routing), not low-level logs.

Evaluate the candidate along three dimensions:

1. Reusable tool-use knowledge or pattern – Does it contain a rule, constraint, mapping, or recovery strategy that can help future function calls?
   - Includes:
     - Tool schema constraints (required fields, enums, type constraints, allowed ranges).
     - Argument-filling strategies (how to infer/ask for missing required args; defaults; disambiguation).
     - Tool selection/routing heuristics (when to use tool A vs B; order of tools).
     - Common failure modes and fixes (400 invalid_request_error, missing tool output, bad JSON, rate limits, retries).
     - Output-parsing templates (how to robustly extract JSON; handle code fences; validate before loads).
     - Multi-turn FC protocol patterns (call_id pairing, tool_call -> tool_output -> next model call).

2. Abstraction level – Is it more than a one-off trace?
   - Prefer:
     - General rules, checklists, invariants, and debugging playbooks.
     - Minimal, canonical examples illustrating the rule.
   - Avoid:
     - Raw transcripts, stack traces, UUIDs, request IDs, or file paths **without** a generalized lesson.
     - Exact tool outputs that are not reusable.
     - “We tried X once” with no stable takeaway.

3. Reliability and stability – Is it correct and likely to remain useful?
   - Prefer:
     - Behaviors required by the FC protocol (e.g., tool_call must have matching tool_output).
     - Constraints directly grounded in tool specs or repeated observations.
   - Avoid:
     - Speculation about model internals.
     - Ephemeral environment quirks (temporary outages) unless generalized as a robust fallback.

Decision rule:
- Default is slightly conservative, but you SHOULD store any candidate that carries at least one **non-trivial, reusable** FC rule or strategy.
- Return `yes` if:
  - The candidate includes some reusable tool-use knowledge/pattern (dimension 1), AND
  - It is not dominated by pure logs/noise, AND
  - It is at least moderately reliable (dimension 3) OR shows clear abstraction (dimension 2).
- Return `no` if the candidate is:
  - Mostly a one-off trace/log with no generalized rule,
  - Extremely local to a single run/filepath/request-id,
  - Or dominated by noise/IDs without a semantic core.

When uncertain BUT the candidate contains at least one meaningful, reusable FC constraint or debugging strategy, prefer answering `yes` rather than `no`.

STRICT OUTPUT: respond with a single lowercase word: `yes` or `no`. Do not explain.

<slot-dump>
{slot_dump}
</slot-dump>
""")

WORKING_SLOT_CHAT_FILTER_USER_PROMPT = dedent("""
You guard Chat-Agent's long-term memory entrance. Decide if this WorkingSlot deserves promotion into FAISS storage.

Your goal is to keep **reusable, personal-context memories** that can help answer future questions about the user's history (e.g., LongMemEval-style temporal and factual questions), not just low-level logs.

Evaluate the slot along three dimensions:

1. Reusable user knowledge or pattern – Does this slot contain stable user information that could be helpful for answering future questions?
   - Includes:
     - User events with clear temporal context (dates, "first time", "before/after" relations).
     - User preferences, habits, or routines that persist over time.
     - User relationships with people, places, organizations, products, or services.
     - Temporal facts that enable timeline reasoning (sequences, durations, orderings).
     - Concrete possessions, activities, or decisions.

2. Abstraction level – Is the content more than ephemeral chatter?
   - Prefer:
     - Specific user facts grounded in session content.
     - Events with temporal anchors (dates, temporal cues).
     - Preferences with clear context or examples.
   - Avoid:
     - Generic assistant advice or explanations (e.g., "Here are some tips...").
     - Pure boilerplate responses with no user-specific information.
     - Vague statements without temporal or factual grounding.

3. Reliability and provenance – Is the information well-supported and traceable?
   - Prefer:
     - Facts with session_id provenance (can trace back to original conversation).
     - Information explicitly stated by the user.
     - Events with specific dates or clear temporal cues.
   - Avoid:
     - Speculation or inferences not grounded in user statements.
     - Information without session provenance.
     - Contradictory or unclear temporal information.

Decision rule:
- Default is slightly conservative, but you SHOULD store any slot that carries at least one **meaningful, reusable** user fact or event.
- Return `yes` if:
  - The slot contains **some reusable user knowledge** (dimension 1), AND
  - It has clear provenance (attachments.session_ids populated), AND
  - It is at least moderately reliable (dimension 3) OR shows clear user-specific content (dimension 2).
- Return `no` if the slot is:
  - Mostly generic assistant responses,
  - Extremely vague with no temporal or factual grounding,
  - Missing session_id provenance,
  - Or dominated by ephemeral chatter with no lasting user information.

When you are uncertain BUT the slot contains at least one meaningful, traceable user fact or event, prefer answering `yes` rather than `no`.

STRICT OUTPUT: respond with a single lowercase word: `yes` or `no`. Do not explain.

<slot-dump>
{slot_dump}
</slot-dump>
""")

WORKING_SLOT_ROUTE_USER_PROMPT = dedent("""
Map this WorkingSlot to the correct ResearchAgent long-term memory family. Choose EXACTLY one label:

- semantic: enduring insights, generalized conclusions, reusable heuristics.
- episodic: Situation → Action → Result traces with metrics, timestamps, or narrative context.
- procedural: step-by-step checklists, reusable commands, or skill blueprints.

Tie-breaking rules:
- Prefer episodic if a chronological action/result trail exists, even if insights appear.
- Otherwise output semantic.

Return only one of: "semantic", "episodic", "procedural".

<slot-dump>
{slot_dump}
</slot-dump>
""")

WORKING_SLOT_COMPRESS_USER_PROMPT = dedent("""
Merge the provided WorkingSlots into ONE distilled WorkingSlot suitable for the short-term queue.

Requirements:
- Remove duplicate facts while keeping supporting metrics or attachments that future agents might need.
- Surface causal links (Situation → Action → Result) whenever present.
- Normalize tags to 1–4 lowercase tokens.
- Keep summary ≤150 words; emphasize reusable, stable insights spanning research, execution, and follow-up actions.
- If attachments include command snippets, metrics, or notes, fold only the most representative subset into the compressed slot.

Input WorkingSlots (JSON):

<slots>
{slots_block}
</slots>

Output format (STRICTLY JSON):
<compressed-slot>
{{
    "stage": "compressed",
    "topic": "concise topic slug",
    "summary": "≤150 words describing the merged knowledge",
    "attachments": {{
        "notes": {{"items": ["bullet 1","bullet 2"]}},
        "metrics": {{"name": value}},
        "procedures": {{"steps": ["step1","step2"]}},
        "artifacts": {{"paths": ["..."]}}
    }},
    "tags": ["tag1","tag2"]
}}
</compressed-slot>
""")

ABSTRACT_EPISODIC_TO_SEMANTIC_PROMPT = dedent("""
You aggregate episodic traces into a single semantic memory entry. Capture the durable lesson that explains why the cluster exists.

Instructions:
- Highlight causal mechanisms, success/failure thresholds, and metrics that repeatedly appeared.
- Mention representative stages (e.g., experiment_execute) only if they add meaning.
- Provide tags that cover both domain concepts and process cues (e.g., ["vision","fog","stability"]).
- Return STRICT JSON containing `summary`, `detail`, `tags`.

Episodic cluster notes:
{episodic_notes}
""")

TRANSFER_SLOT_TO_TEXT_PROMPT = dedent("""
Convert the WorkingSlot JSON into a concise human-readable paragraph (no tags, no JSON). This summary feeds chat surfaces, not FAISS.

Guidance:
- Mention stage, topic, and the core outcome or decision.
- Cite standout metrics or attachments inline (e.g., "accuracy climbed to 0.73").
- Describe actionable next steps only if explicitly recorded.
- Limit to 2–4 sentences; avoid bulleting or markdown.

Input WorkingSlot (JSON):

{dump_slot_json}
""")

TRANSFER_QA_AGENT_CONTEXT_TO_WORKING_SLOT_PROMPT = dedent("""
Convert the QA Agent workflow context into at most {max_slots} WorkingSlot and at least 1 WorkingSlot entries ready for filtering/routing.

Goal:
- Only create slots that are worth storing for long-term reuse.
- A slot SHOULD be created if and only if it is:
  (A) Semantic evidence: stable, factual content (e.g., retrieved passages, environment observations) that can later support answering other questions.
  OR
  (B) Episodic experience: reusable process knowledge, strategy, or failure pattern about how to solve HotpotQA-style multi-hop questions.

Ignore:
- Ephemeral chatter, low-level execution logs, or transient chain-of-thought that does not generalize.
- Speculation not grounded in retrieved content or clear experience.

Context Snapshot:
<workflow-context>
{snapshot}
</workflow-context>

Authoring rules:
1. Each slot MUST capture a single reusable takeaway (one decision, discovery, bottleneck, or command).
2. `stage` MUST be one of: question_understanding, information_retrieval, answer_generation, answer_validation, meta.
   - Use `information_retrieval` / `answer_validation` often for semantic evidence.
   - Use `question_understanding` / `answer_generation` / `meta` often for episodic experience.
3. `summary` follows Situation → Action → Result whenever data exists; keep ≤80 words and make it self-contained.
4. `topic` is a 3–6 word slug referencing the problem space (lowercase, space-separated), e.g. "bridge entity retrieval", "evidence aggregation failure".
5. `attachments` is optional but, when present, group similar info under keys such as:
    - "notes": {{"items": []}}        # short bullet-like notes or paraphrased facts
    - "references": {{"links": []}}   # source titles, IDs, or URLs (e.g. Wikipedia pages)
    - "issues": {{"list": []}}       # open problems, errors, or caveats
    - "actions": {{"list": []}}       # next steps or commands
6. `tags` is a list of lowercase keywords (≤5 items) mixing:
    - domain hints: "hotpotqa","wikipedia","multi-hop","bridge-entity"
    - workflow hints: "semantic-evidence","episodic-experience","retrieval","planning","verification","failure"
   Use "semantic-evidence" for factual slots and "episodic-experience" for process/strategy slots.

Output STRICTLY as JSON within the tags below (no extra commentary):
{{
    "slots": [
        {{
            "stage": "information_retrieval",
            "topic": "bridge entity localization",
            "summary": "Situation/Action/Result narrative focusing on a single reusable semantic evidence or episodic pattern.",
            "attachments": {{
                "notes": {{"items": ["short fact or note 1", "short fact or note 2"]}},
                "references": {{"links": ["wikipedia:Albert_Einstein"]}},
                "issues": {{"list": []}},
                "actions": {{"list": ["optional follow-up"]}}
            }},
            "tags": ["hotpotqa","semantic-evidence","retrieval"]
        }}
    ]
}}
""")

TRANSFER_FC_AGENT_CONTEXT_TO_WORKING_SLOT_PROMPT = dedent("""
Convert the FC Agent workflow context into at most {max_slots} WorkingSlot entries ready for filtering/routing.

BFCL characteristics to account for:
- Multiple candidate tools may be present; only some are relevant.
- Required parameters may be missing; safe behavior is to ask for clarification (not guess).
- Multi-step: tool outputs in earlier steps constrain later arguments; retries are common.
- Some system messages explicitly forbid assumptions; preserve such constraints as evidence.
- Tool descriptions / schemas can be noisy; store disambiguation rules as procedural experience.

PRIMARY GOAL (IMPORTANT):
Store only **reusable, retrieval-worthy** memories that improve future tool-use accuracy.
Prefer:
1) PROCEDURAL playbooks (tool-agnostic checklists you can reuse),
2) SEMANTIC invariants (validated constraints / schema rules / arg mappings grounded in evidence),
and only then EPISODIC lessons (rare failure modes not expressible as a generic SOP).

CRITICAL QUALITY BAR (MOST IMPORTANT):
- If the snapshot does NOT contain any **novel, validated** takeaway, output ZERO slots (empty list).
- DO NOT generate generic boilerplate SOPs ("validate required fields", "confirm enums") unless the snapshot provides a specific, non-obvious constraint, mapping, or failure/fix that would change behavior next time.
- Prefer 1–3 high-signal slots over many low-signal slots.
- Do NOT store raw chain-of-thought or verbose inner reasoning.

SUCCESS-PATH PRIORITY (VERY IMPORTANT):
- Prefer takeaways from steps that are **ultimately used** or **validated** by tool schema / tool outputs / final answer.
- Do NOT store intermediate hypotheses or attempts that are contradicted later in the snapshot.
- Failures may be stored ONLY as "symptom → likely cause → reusable fix", and must include a clear trigger/condition.

EVIDENCE & GROUNDEDNESS (VERY IMPORTANT):
- Every slot must cite evidence from the snapshot via attachments:
  - semantic: evidence via "tool_schema"/"observations"/"constraints"/"arg_map"
  - procedural: evidence via "checks"/"failures" (if any) + a grounded playbook
- If you cannot ground a claim in the snapshot (schema text, tool output, explicit constraint), do NOT store it.

ALLOWLIST OF SPECIFIC TOKENS (IMPORTANT):
- You are encouraged to preserve **non-private** schema tokens when decisive:
  - required field names, enum literals, tool names, tool call/response protocol tokens.
- Do NOT include private IDs (addresses, account numbers). If needed, anonymize them.

DEDUPLICATION (IMPORTANT):
- If two candidate slots would be near-duplicates, keep only the more general and more evidenced one.
- Each slot must capture EXACTLY ONE reusable takeaway.

OUTPUT SIZE POLICY:
- Target 1–3 slots.
- You may output 0 slots.
- Only output >3 slots if the snapshot contains multiple distinct validated constraints AND at least one rare failure/fix.

Context Snapshot (may include dialogue history, tool schemas, tool outputs):
<bfcl-context>
{snapshot}
</bfcl-context>

Authoring rules:
1. Each slot MUST capture exactly ONE reusable takeaway.
2. `stage` MUST be one of:
   - intent_constraints         # user intent + hard constraints (units, style, "no assumptions", etc.)
   - tool_selection             # selecting the right tool among distractors
   - argument_construction      # mapping text -> required args, enums, defaults policy
   - tool_execution             # calling tools, handling returned outputs
   - result_integration         # merging tool outputs into next-turn state/answer
   - error_handling             # retries, validation failures, unsupported protocol, etc.
   - meta                       # evaluation protocol / agent-control insights
3. `summary` MUST be ≤90 words, self-contained, and use the format by memory type:
   - Procedural (preferred): Goal → Preconditions → Steps → Checks (compact, imperative).
   - Semantic: Invariant/Constraint → Evidence → Implication.
   - Episodic (rare): Situation → Action → Result → Fix/Generalization.
4. `topic` is a 3–7 word slug (lowercase, space-separated),
   e.g. "no-assumption arg filling", "tool distractor disambiguation".
5. `attachments` is optional but, when present, use these keys when relevant:
   - "constraints": {{"items": []}}      # explicit do/don't rules from snapshot
   - "tool_schema": {{"items": []}}      # compact schema notes: required fields, enums, defaults policy
   - "arg_map": {{"items": []}}          # text-to-arg mapping patterns grounded in snapshot
   - "observations": {{"items": []}}     # key tool outputs / environment facts (paraphrased)
   - "failures": {{"items": []}}         # error symptoms + likely causes (only if seen)
   - "recovery": {{"steps": []}}         # reusable playbook steps (3–7), grounded and specific
   - "checks": {{"items": []}}           # validation checks before/after calling tools
6. For PROCEDURAL slots:
   - Include "recovery": {{"steps": [...]}} as a reusable SOP (3–7 steps).
   - Steps should be tool-agnostic where possible, but may mention specific schema tokens when decisive.
   - If the snapshot provides no specific evidence beyond generic best practice, DO NOT output a procedural slot.
7. For SEMANTIC slots:
   - Prefer filling "constraints"/"tool_schema"/"arg_map"/"observations" with compact bullets grounded in snapshot.
8. For EPISODIC slots:
   - Keep them rare; only for a novel failure mode or tactic NOT expressible as a generic SOP.
   - Must include a reusable fix and a clear trigger condition.

Routing hints (implicit, do not add extra fields beyond schema):
- semantic: stable constraints, schemas, validated outputs, invariant mappings.
- procedural: reusable step-by-step playbooks with checks.
- episodic: rare strategies/failures that generalize but cannot be expressed as a generic SOP.

Output STRICTLY as JSON within the tags below (no extra commentary):
{{
  "slots": [
    {{
      "stage": "argument_construction",
      "topic": "no-assumption required-args protocol",
      "summary": "Goal: produce a valid tool call without guessing. Preconditions: required fields missing in user text and no defaults stated in schema. Steps: enumerate required fields; ask a single clarification listing only missing fields; restate confirmed args; fill only schema-allowed values; construct the call. Checks: all required fields present; enum literals match schema; no conflicting constraints. Result: avoids invalid requests and respects no-assumption rules.",
      "attachments": {{
        "constraints": {{"items": ["do not guess missing required args; ask targeted clarification"]}},
        "tool_schema": {{"items": ["required: field_a, field_b", "enum field_c: [x,y,z]"]}},
        "recovery": {{"steps": ["extract required fields and enums from schema", "diff against user-provided info", "ask one clarification listing missing fields only", "validate enum/value constraints", "construct call using only confirmed fields", "re-check required fields and conflicts before sending"]}},
        "checks": {{"items": ["all required fields present", "enum literals match schema", "no conflicting constraints"]}}
      }},
      "tags": ["bfcl","function-calling","procedural-experience","arg-filling","clarification","validation"]
    }}
  ]
}}
""")

TRANSFER_CHAT_AGENT_CONTEXT_TO_WORKING_SLOT_PROMPT = dedent("""
Convert the Chat Agent workflow context into at most {max_slots} WorkingSlot entries ready for filtering/routing.

This snapshot is GUARANTEED to correspond to EXACTLY ONE unique session_id.

========================
HARD CONSTRAINTS (MUST)
========================
1) SINGLE SESSION ID (NO GUESSING):
- You MUST locate the single unique Session ID in the snapshot.
- NEVER fabricate session IDs. NEVER use placeholders like "unknown_session".

2) AT LEAST ONE SLOT IF SESSION ID EXISTS:
- If and only if a valid single Session ID is found, you MUST output AT LEAST 1 slot.
- If no storeworthy user/assistant action or preference exists, output ONE fallback slot:
  - stage="meta"
  - topic like "no storeworthy info in session"
  - summary states: no concrete user/assistant actions/preferences found in this session
  - Still include required attachments with the real session_id

3) SESSION ID MUST BE PRESENT IN EVERY SLOT:
- Every produced slot MUST include attachments.session_ids.items == [THE_SESSION_ID].
- Do NOT output any slot without session_id in attachments.
- Do NOT include any other session ids.

========================
WHAT TO EXTRACT (NOT mutually exclusive)
========================
A) USER ACTIONS / EXPERIENCES -> episodic slots (user_event or user_timeline)
B) USER PREFERENCES / STABLE TASTES -> user_preference slots (semantic-evidence)
C) ASSISTANT ACTIONS that materially affected outcome -> episodic slots (user_event or meta)

========================
Context Snapshot:
<chat-context>
{snapshot}
</chat-context>

Authoring rules:
1. Output 1..{max_slots} slots IF a valid single session_id exists.
2. Each slot MUST capture a single action/event/timeline OR a single inferred preference OR a single assistant-action episode.
3. `stage` MUST be one of: user_event, user_timeline, user_preference, user_relationship, meta
4. `summary` MUST be ≤80 words and action-focused.
5. `topic` is a 3–6 word slug (lowercase, space-separated).
6. `attachments` is REQUIRED and MUST include ALL keys:
   - "session_ids": {{"items": [THE_SESSION_ID]}}  # REQUIRED, singleton
   - "dates": {{"items": []}}                      # if no explicit date, use ["date not specified"]
   - "entities": {{"items": []}}
   - "facts": {{"items": []}}                      # NEVER empty
   - "temporal_cues": {{"items": []}}              # if no cues, use ["timing not specified"]
7. For episodic slots, `dates` and `temporal_cues` MUST be non-empty (use the fallback strings above).

`tags` policy:
- tags ≤6 lowercase.
- Must include either "episodic-experience" (events/timelines/assistant-actions) or "semantic-evidence" (preferences).
- Add "assistant-action" tag when describing assistant actions.

Output STRICTLY as JSON (no extra commentary):
{{
  "slots": [
    {{
      "stage": "meta",
      "topic": "no storeworthy info in session",
      "summary": "No concrete user/assistant actions or stable preferences were found in this session; recorded as a coverage marker.",
      "attachments": {{
        "session_ids": {{"items": ["THE_SESSION_ID"]}},
        "dates": {{"items": ["date not specified"]}},
        "entities": {{"items": ["user", "assistant"]}},
        "facts": {{"items": ["no storeworthy user/assistant action/preference detected in this session"]}},
        "temporal_cues": {{"items": ["timing not specified"]}}
      }},
      "tags": ["episodic-experience","meta","assistant-action"]
    }}
  ]
}}
""")

TRANSFER_EXPERIMENT_AGENT_CONTEXT_TO_WORKING_SLOTS_PROMPT = dedent("""
Convert the Experiment Agent workflow context into at most {max_slots} WorkingSlot entries ready for filtering/routing.

Context Snapshot:
<workflow-context>
{snapshot}
</workflow-context>

Authoring rules:
1. Each slot MUST capture a single reusable takeaway (decision, discovery, bottleneck, or command).
2. `stage` MUST be one of: pre_analysis, code_plan, code_implement, code_judge, experiment_execute, experiment_analysis, meta.
3. `summary` follows Situation → Action → Result whenever data exists; keep ≤130 words.
4. `topic` is a 3–6 word slug referencing the problem space.
5. `attachments` is optional but, when present, group similar info under keys such as
   - "notes": {{"items": []}}
   - "metrics": {{}}
   - "issues": {{"list": []}}
   - "actions": {{"list": []}}
6. `tags` is a list of lowercase keywords (≤5 items) mixing domain + workflow hints.
7. If the context lacks meaningful content, return `"slots": []` but keep the envelope.

Output STRICTLY as JSON within the tags below:
{{
    "slots": [
    {{
        "stage": "code_plan",
        "topic": "coverage planning",
        "summary": "Situation/Action/Result narrative...",
        "attachments": {{
            "notes": {{"items": ["detail 1", "detail 2"]}},
            "metrics": {{"acc": 0.92}},
            "issues": {{"list": []}},
            "actions": {{"list": ["follow-up 1"]}}
        }},
        "tags": ["plan","coverage"]
    }}
    ]
}}
""")

TRANSFER_SLOT_TO_SEMANTIC_RECORD_PROMPT_EXPEIRMENT = dedent("""
Transform the WorkingSlot into a semantic memory entry suitable for FAISS retrieval in HotpotQA-style multi-hop QA.

Expectations:
- The semantic record MUST capture **factual evidence** grounded in the WorkingSlot (e.g., retrieved passages, environment observations), not planning logic or agent-control flow.
- `summary` (≤80 words) is a compact, question-agnostic factual statement or tightly related fact cluster that can be reused as evidence (e.g., key relations, attributes, dates, locations).
- `detail` elaborates the supporting evidence: paraphrased or briefly quoted spans, source/page titles or IDs, and important caveats. Use "\\n" to separate logically distinct atomic facts or evidence items.
- Avoid speculation or heuristic advice; only include information that is directly supported by the WorkingSlot content.
- `tags` should mix entity names, domain hints, and relation/type hints (e.g., ["hotpotqa","wikipedia","albert-einstein","birthplace"]).

<working-slot>
{dump_slot_json}
</working-slot>

Output STRICTLY as JSON inside the tags, with no extra text:
<semantic-record>
{{
    "summary": "semantic evidence summary",
    "detail": "expanded factual evidence and context",
    "tags": ["keyword1","keyword2"]
}}
</semantic-record>
""")

TRANSFER_SLOT_TO_SEMANTIC_RECORD_PROMPT_QA = dedent("""
Transform the WorkingSlot into a semantic memory entry suitable for FAISS retrieval in HotpotQA-style multi-hop QA.

Expectations:
- The semantic record MUST capture **factual evidence** grounded in the WorkingSlot (e.g., retrieved Wikipedia passages, entity attributes, relations).
- `summary` (≤80 words) is a compact, question-agnostic factual statement that can be reused as evidence for future multi-hop questions.
- `detail` elaborates the supporting evidence:
  - Paraphrased or briefly quoted facts from retrieved sources.
  - Entity relations (A located_in B, A spouse_of B, A part_of B, etc.).
  - Source references (Wikipedia page titles, IDs) when available.
  - Use "\\n" to separate logically distinct atomic facts.
- Avoid speculation or strategy advice; only include information directly supported by the WorkingSlot content.
- `tags` should mix:
  - Entity names: "albert-einstein", "new-york-city"
  - Domain hints: "hotpotqa", "wikipedia", "multi-hop"
  - Relation types: "birthplace", "spouse", "location", "part-of"

<working-slot>
{dump_slot_json}
</working-slot>

Output STRICTLY as JSON inside the tags, with no extra text:
<semantic-record>
{{
    "summary": "Compact factual evidence (e.g., 'Albert Einstein was born in Ulm, Germany on March 14, 1879. He received Nobel Prize in Physics in 1921.')",
    "detail": "Atomic fact 1: Einstein birthplace is Ulm, Germany.\\nAtomic fact 2: Birth date March 14, 1879.\\nAtomic fact 3: Nobel Prize Physics 1921.\\nSource: Wikipedia:Albert_Einstein",
    "tags": ["hotpotqa","wikipedia","albert-einstein","birthplace","nobel-prize"]
}}
</semantic-record>
""")

TRANSFER_SLOT_TO_SEMANTIC_RECORD_PROMPT_CHAT = dedent("""
Transform the WorkingSlot into a semantic memory entry suitable for FAISS retrieval in LongMemEval-style personal chat history.

Expectations:
- The semantic record MUST capture **stable user facts** grounded in the WorkingSlot (preferences, attributes, relationships, possessions).
- `summary` (≤80 words) is a compact, context-independent factual statement about the user.
- `detail` elaborates the evidence:
  - User attributes or preferences stated across sessions.
  - Relationships with people, places, organizations.
  - Possession or usage of products/services.
  - Include session_id for provenance.
  - Use "\\n" to separate distinct facts.
- Avoid event narratives (those go to episodic); focus on **timeless facts**.
- `tags` should mix:
  - Domain: "car", "travel", "finance", "health", "shopping"
  - Fact type: "preference", "attribute", "relationship", "possession"
  - Memory type: "semantic-evidence", "user-preference"

<working-slot>
{dump_slot_json}
</working-slot>

Output STRICTLY as JSON inside the tags, with no extra text:
<semantic-record>
{{
    "summary": "Compact user fact (e.g., 'User owns a new car, gets regular service at dealership, uses Shell gas station rewards program.')",
    "detail": "Fact 1: User owns new car (mentioned in session answer_4be1b6b4_2).\\nFact 2: Prefers dealership for service.\\nFact 3: Active Shell rewards program member.\\nFact 4: Tracks gas mileage (avg 32 mpg).\\nProvenance: sessions answer_4be1b6b4_2, answer_4be1b6b4_3",
    "tags": ["car","user-preference","semantic-evidence","rewards-program"]
}}
</semantic-record>
""")

TRANSFER_SLOT_TO_SEMANTIC_RECORD_PROMPT_FC = dedent("""
Transform the WorkingSlot into a semantic memory entry suitable for FAISS retrieval in BFCL-style function calling.

Expectations:
- The semantic record MUST capture **stable function-calling knowledge** (tool schemas, constraints, argument mappings).
- `summary` (≤80 words) is a compact, reusable constraint or invariant about tool usage.
- `detail` elaborates the evidence:
  - Tool schema details (required fields, enums, type constraints).
  - Argument-filling constraints (no-assumption rules, defaults policy).
  - Validated input-output mappings.
  - Include specific tool names and field names when decisive.
  - Use "\\n" to separate distinct rules.
- Avoid procedural steps (those go to procedural); focus on **declarative knowledge**.
- `tags` should mix:
  - Tool names: "get_weather", "book_flight"
  - Constraint types: "required-field", "enum-constraint", "no-assumption"
  - Domain: "bfcl", "function-calling", "semantic-evidence"

<working-slot>
{dump_slot_json}
</working-slot>

Output STRICTLY as JSON inside the tags, with no extra text:
<semantic-record>
{{
    "summary": "Compact tool constraint (e.g., 'get_weather requires location (enum: city names) and units (enum: celsius/fahrenheit). No defaults; must ask user if missing.')",
    "detail": "Tool: get_weather\\nRequired: location (type: enum[city_names]), units (type: enum[celsius, fahrenheit])\\nConstraint: no assumption rule - ask clarification if either field missing\\nEnum validation: units must match exactly 'celsius' or 'fahrenheit'\\nSource: tool schema + no-assumption system message",
    "tags": ["bfcl","function-calling","get-weather","required-field","enum-constraint","no-assumption"]
}}
</semantic-record>
""")

TRANSFER_SLOT_TO_EPISODIC_RECORD_PROMPT_EXPRIMENT = dedent("""
Convert the WorkingSlot into an episodic memory record emphasizing Situation → Action → Result.

<working-slot>
{dump_slot_json}
</working-slot>

Output STRICTLY as JSON inside the tags:
<episodic-record>
{{
    "stage": "{stage}",
    "summary": "≤80 word Situation → Action → Result overview",
    "detail": {{
        "situation": "Context and constraints",
        "actions": ["action 1","action 2"],
        "results": ["result 1","result 2"],
        "metrics": {{}},
        "artifacts": []
    }},
    "tags": ["keyword1","keyword2"]
}}
</episodic-record>
""")

TRANSFER_SLOT_TO_EPISODIC_RECORD_PROMPT_QA = dedent("""
Convert the WorkingSlot into an episodic memory record emphasizing the QA process: Question → Retrieval/Reasoning → Answer.

Expectations:
- The episodic record captures **reusable QA patterns or strategies** that worked (or failed) for multi-hop questions.
- `summary` (≤80 words) follows: Question Type → Strategy → Result.
- `detail` elaborates:
  - "situation": Question characteristics, entities involved, required reasoning type.
  - "actions": Retrieval steps, reasoning steps, bridge entity identification.
  - "results": Answer found/not found, what worked/failed.
  - "observations": Key insights about question pattern or retrieval strategy.
- Focus on **generalizable patterns**, not one-off traces.
- `tags` should mix:
  - Question type: "bridge-entity", "comparison", "temporal-reasoning"
  - Strategy: "two-hop-retrieval", "entity-expansion"
  - Outcome: "success", "failure", "partial"

<working-slot>
{dump_slot_json}
</working-slot>

Output STRICTLY as JSON inside the tags:
<episodic-record>
{{
    "stage": "{stage}",
    "summary": "≤80 word Question Type → Strategy → Result narrative (e.g., 'Bridge entity question about birthplace and university. Strategy: retrieve person's birthplace first, then search university in that city. Result: successfully found answer.')",
    "detail": {{
        "situation": "Multi-hop question requiring bridge entity (person → birthplace → university)",
        "actions": ["retrieved person's Wikipedia page", "extracted birthplace entity", "searched universities in birthplace city", "verified university founding date"],
        "results": ["answer found: University of Ulm", "bridge entity strategy effective"],
        "observations": ["birthplace serves as reliable bridge for person-location questions", "Wikipedia infoboxes contain structured relation data"]
    }},
    "tags": ["hotpotqa","bridge-entity","two-hop","episodic-experience","success"]
}}
</episodic-record>
""")

TRANSFER_SLOT_TO_EPISODIC_RECORD_PROMPT_CHAT = dedent("""
Convert the WorkingSlot into an episodic memory record suitable for FAISS retrieval in LongMemEval-style personal chat history.

**CRITICAL**: You MUST populate ALL fields in the output, even if some attachments are empty. Use the WorkingSlot's `summary`, `topic`, and available `attachments` to infer and reconstruct the event details.

Expectations:
- The episodic record MUST capture the **user's action or experience** described in the WorkingSlot.
- `summary` (≤80 words): Rewrite the slot's summary into a narrative: What the user DID → When (if known) → Context → Outcome.
  - If no explicit date, use "at some point" or "during a conversation" as temporal anchor.
- `detail` MUST be fully populated using inference from the slot content:
  - "session_id": MUST be extracted from `attachments.session_ids.items[0]`.
  - "situation": Infer from `topic` and `summary` - what was the user's context/goal?
  - "actions": Extract or infer user actions from `summary` and `attachments.facts`. NEVER leave empty - at minimum, paraphrase the summary as an action.
  - "results": Infer outcomes from `summary`. If no explicit outcome, state "outcome not specified" or infer likely result.
  - "temporal_context": 
    - "dates": Copy from `attachments.dates.items`. If empty, use `["date not specified"]`.
    - "temporal_cues": Copy from `attachments.temporal_cues.items`. If empty, use `["timing not specified"]`.
    - "sequence": Describe event order if multiple actions, otherwise "single event".
  - "entities_involved": Copy from `attachments.entities.items`. If empty, extract key nouns from summary.
  - "facts": Copy from `attachments.facts.items`. If empty, extract key facts from summary.

**INFERENCE RULES (IMPORTANT)**:
- If `attachments.dates` is empty: set `temporal_context.dates` to `["date not specified"]`, NOT empty list.
- If `attachments.temporal_cues` is empty: set `temporal_context.temporal_cues` to `["timing not specified"]`, NOT empty list.
- If `attachments.facts` is empty: extract facts from the `summary` field.
- If `attachments.entities` is empty: extract entity names from `summary` and `topic`.
- NEVER return empty arrays `[]` for actions, results, entities_involved, or facts.

<working-slot>
{dump_slot_json}
</working-slot>

Output STRICTLY as JSON inside the tags (ALL fields must be non-empty):
<episodic-record>
{{
    "stage": "{stage}",
    "summary": "Rewritten narrative: User [ACTION] [WHEN if known] [CONTEXT] [OUTCOME]. Example: 'User sought band recommendations similar to The Electric Storm after attending their concert at a music festival. Looking for similar indie rock artists.'",
    "detail": {{
        "session_id": "from attachments.session_ids.items[0], e.g., '7045db85_1'",
        "situation": "Inferred context/goal, e.g., 'User attended The Electric Storm concert, enjoyed it, wanted to discover similar music'",
        "actions": ["NEVER EMPTY - at minimum paraphrase summary, e.g., 'attended The Electric Storm concert', 'asked for similar band recommendations'"],
        "results": ["Inferred or stated outcomes, e.g., 'received recommendations', 'outcome not explicitly stated'"],
        "temporal_context": {{
            "dates": ["from attachments or 'date not specified'"],
            "temporal_cues": ["from attachments or 'timing not specified'"],
            "sequence": "event order description or 'single event'"
        }},
        "entities_involved": ["NEVER EMPTY - extract from summary/topic, e.g., 'user', 'The Electric Storm', 'Music Festival'"],
        "facts": ["NEVER EMPTY - from attachments.facts or extract from summary, e.g., 'user attended Electric Storm concert', 'user seeking similar band recommendations'"]
    }},
    "tags": ["event-type", "domain", "episodic-experience"]
}}
</episodic-record>
""")

TRANSFER_SLOT_TO_EPISODIC_RECORD_PROMPT_FC = dedent("""
Convert the WorkingSlot into an episodic memory record emphasizing the function-calling process: Intent → Tool Selection → Execution → Result.

Expectations:
- The episodic record captures **rare, reusable FC failure/recovery patterns**, not generic playbooks.
- `summary` (≤80 words) follows: Situation (trigger) → Action (what was tried) → Result (outcome) → Fix/Learning.
- `detail` elaborates:
  - "situation": User intent, available tools, constraints, trigger condition for this episode.
  - "actions": Tool selection decisions, argument construction attempts, execution steps.
  - "results": Success/failure, error messages, tool outputs.
  - "fix": How the issue was resolved (if failure), generalized lesson.
  - "trigger": Clear condition when this pattern applies.
- Focus on **novel failure modes** or **non-obvious tactics**, not common best practices.
- `tags` should mix:
  - Stage: "tool-selection", "argument-construction", "error-handling"
  - Outcome: "failure-recovered", "success-pattern"
  - Tool-specific identifiers if relevant

<working-slot>
{dump_slot_json}
</working-slot>

Output STRICTLY as JSON inside the tags:
<episodic-record>
{{
    "stage": "{stage}",
    "summary": "≤80 word Situation → Action → Result → Fix narrative (e.g., 'Ambiguous location input triggered tool selection failure. Tried city-based get_weather, got 400 error. Fix: ask user to clarify between city name vs zip code, then route to correct tool variant.')",
    "detail": {{
        "situation": "User said 'weather in springfield' - ambiguous (multiple cities named Springfield)",
        "actions": ["attempted get_weather_by_city with 'springfield'", "received 400 invalid_request_error: ambiguous location"],
        "results": ["initial call failed", "error message: 'multiple matches for springfield'"],
        "fix": "Ask clarification: 'Which Springfield? (IL, MA, MO, etc.)', then use get_weather_by_city with state qualifier",
        "trigger": "Location input matches multiple cities; tool returns ambiguous location error"
    }},
    "tags": ["bfcl","function-calling","error-handling","ambiguous-input","episodic-experience"]
}}
</episodic-record>
""")


TRANSFER_SLOT_TO_PROCEDURAL_RECORD_PROMPT_EXPERIMENT = dedent("""
Convert the WorkingSlot into a procedural memory entry that captures a reusable skill or checklist.

<working-slot>
{dump_slot_json}
</working-slot>

Output STRICTLY as JSON inside the tags:
<procedural-record>
{{
    "name": "short skill name",
    "description": "≤60 words explaining when/why to apply it",
    "steps": ["step 1","step 2","step 3"],
    "code": "optional snippet or empty string",
    "tags": ["keyword1","keyword2"]
}}
</procedural-record>
""")

TRANSFER_SLOT_TO_PROCEDURAL_RECORD_PROMPT_QA = dedent("""
Convert the WorkingSlot into a procedural memory entry that captures a reusable QA strategy or checklist.

Expectations:
- The procedural record captures **step-by-step playbooks** for solving specific types of multi-hop questions.
- `name`: Short, descriptive skill name (e.g., "bridge entity two-hop retrieval").
- `description` (≤60 words): When/why to apply this strategy.
- `steps`: Ordered, actionable steps (3-7 items) that are reusable across similar questions.
- `code`: Optional pseudo-code or empty string.
- `tags`: Question types this applies to.

<working-slot>
{dump_slot_json}
</working-slot>

Output STRICTLY as JSON inside the tags:
<procedural-record>
{{
    "name": "bridge entity two-hop retrieval",
    "description": "Use when question requires connecting two entities through an intermediate bridge entity. Common for 'Where was X born' + 'What university in that city' patterns.",
    "steps": [
        "identify the bridge entity from question (often a person or location)",
        "retrieve Wikipedia page for bridge entity",
        "extract the linking attribute (birthplace, location, affiliation)",
        "search for target entity using the linking attribute as constraint",
        "verify target entity meets all question criteria"
    ],
    "code": "",
    "tags": ["hotpotqa","bridge-entity","two-hop","procedural-experience"]
}}
</procedural-record>
""")

TRANSFER_SLOT_TO_PROCEDURAL_RECORD_PROMPT_CHAT = dedent("""
Convert the WorkingSlot into a procedural memory entry that captures a reusable pattern for handling personal chat memory.

Expectations:
- The procedural record captures **strategies for extracting, organizing, or reasoning over user's personal information**.
- `name`: Short, descriptive skill name (e.g., "timeline reconstruction from scattered events").
- `description` (≤60 words): When/why to apply this strategy in personal chat context.
- `steps`: Ordered, actionable steps (3-7 items) for handling similar user information patterns.
- `code`: Optional pseudo-code or empty string.
- `tags`: User information types this applies to.

<working-slot>
{dump_slot_json}
</working-slot>

Output STRICTLY as JSON inside the tags:
<procedural-record>
{{
    "name": "timeline reconstruction from events",
    "description": "Use when user mentions multiple related events across sessions. Reconstruct chronological sequence to answer 'when/before/after/how long' questions about user's personal history.",
    "steps": [
        "extract all events related to the topic from attachments.facts",
        "collect dates and temporal_cues for each event",
        "order events chronologically using dates and relative cues",
        "identify causal or dependency relationships between events",
        "construct timeline sequence with explicit ordering",
        "verify consistency with session_ids for provenance"
    ],
    "code": "",
    "tags": ["chat","user-timeline","temporal-reasoning","procedural-experience"]
}}
</procedural-record>
""")

TRANSFER_SLOT_TO_PROCEDURAL_RECORD_PROMPT_FC = dedent("""
Convert the WorkingSlot into a procedural memory entry that captures a reusable function-calling playbook.

Expectations:
- The procedural record captures **step-by-step checklists** for common FC tasks (argument filling, validation, error recovery).
- `name`: Short, descriptive skill name (e.g., "no-assumption required-args protocol").
- `description` (≤60 words): When/why to apply this protocol.
- `steps`: Ordered, actionable steps (3-7 items) that are tool-agnostic where possible.
  - May mention specific schema tokens when decisive.
  - Must be grounded in evidence from the WorkingSlot.
- `code`: Optional pseudo-code or empty string.
- `tags`: FC stages and patterns this applies to.

<working-slot>
{dump_slot_json}
</working-slot>

Output STRICTLY as JSON inside the tags:
<procedural-record>
{{
    "name": "no-assumption required-args protocol",
    "description": "Apply when tool schema has required fields and system message forbids guessing. Ensures all required args are user-confirmed before calling tool.",
    "steps": [
        "extract required fields and enums from tool schema",
        "diff required fields against user-provided information",
        "if any required field missing: ask single clarification listing only missing fields",
        "validate enum/value constraints against schema",
        "construct tool call using only confirmed fields",
        "re-check all required fields present and no conflicting constraints before sending"
    ],
    "code": "",
    "tags": ["bfcl","function-calling","procedural-experience","arg-filling","no-assumption","validation"]
}}
</procedural-record>
""")