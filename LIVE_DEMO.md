# Main-page live demonstration

The normal console sends only the question to `/query`. Both initial and final
answers are computed by the running system. The main page has no injected-mode
selector, stored faulty starting answers, or preset correct outputs.

## Questions checked on the current T4 session, 28 October 2026

| Question | Actual initial answer | Actual final answer | Final signal status |
|---|---|---|---|
| Which college president of Notre Dame is credited with preventing more confrontations between students and the KKK? | I don't have enough information to answer this question. | Fr. Matthew Walsh | supported |
| In what year did the student paper Common Sense begin publication at Notre Dame? | I don't have enough information to answer this question. | 1987 | unverified |
| In what year was the Main Building at Notre Dame razed in a fire? | 1879 | 1879 | supported |
| What is the primary seminary of the Congregation of the Holy Cross? | Moreau Seminary | Moreau Seminary | supported |

Start with the **College president** preset for an example of genuine recovery.
The main page features only **College president** and **Publication year**, the
two normal questions where recovery replaced an initial refusal with an answer.
The other records above remain in this inspection log but are omitted from the
main-page example list.
The source passages support the listed factual answers. The 1987 answer remained
unverified under the configured signal checks; the UI preserves that status.

These are normal live checks, separate from the earlier 138-case injected-input
evaluation. Ten questions were checked in the current session. Four ended in
refusal, including two cases whose initial answers were correct. This is a known
response-selection limitation; the checks are not a new aggregate accuracy claim.
The BERT router selected retrieval failure for all ten normal questions. No
classifier retraining, generation changes, reference-answer injection or benchmark
rewriting was performed for this interface update.

Full query results are saved in `demo_validation/live_normal_questions_20261028.json`
in the active Colab project and persisted to its mounted Drive project. This file
is an inspection record, not an inference lookup table. Repeated live answers can
vary with the runtime and model environment.
