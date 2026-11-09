/* Selected actual normal-query records; never used as inference responses. */
window.RAG_LIVE_EXAMPLES = {
  "date": "2026-10-28",
  "scope": "Selected normal live POST /query checks on Colab T4; no injected inputs",
  "note": "Selected live runs checked on 28 October 2026. Run live question uses fresh normal retrieval and generation; repeated results can differ. The full benchmark below is unchanged.",
  "examples": [
    {
      "query": "Which college president of Notre Dame is credited with preventing more confrontations between students and the KKK?",
      "initial_answer": "I don't have enough information to answer this question.",
      "final_answer": "Fr. Matthew Walsh",
      "outcome": "supported"
    },
    {
      "query": "In what year did the student paper Common Sense begin publication at Notre Dame?",
      "initial_answer": "I don't have enough information to answer this question.",
      "final_answer": "1987",
      "outcome": "unverified"
    }
  ]
};
