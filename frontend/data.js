// Example document pairs for the Compare tab.
export const PRESETS = [
  {
    id: 'para',
    label: 'Paraphrase',
    a: 'The government announced new measures to reduce air pollution in major cities. Factories will have to install cleaner filters by next year. Citizens are encouraged to use public transport.',
    b: 'Authorities revealed fresh steps to cut smog in large urban areas. Industrial plants must fit better filtration systems within twelve months. People are urged to take buses and trains instead of driving.',
  },
  {
    id: 'plag',
    label: 'Plagiarism',
    a: 'TF-IDF is a fast and interpretable baseline for measuring document similarity. It weights each term by how rare it is across the corpus. However, it ignores word order and context.',
    b: 'TF-IDF is a quick and interpretable baseline for comparing document similarity. It weights every term by how uncommon it is across the whole corpus. But it ignores context and word order.',
  },
  {
    id: 'unrel',
    label: 'Unrelated',
    a: 'Preheat the oven and bake the bread for forty minutes until golden. Let it cool before slicing.',
    b: 'The batsman scored a century as India won the cricket match by six wickets. The crowd cheered loudly.',
  },
  {
    id: 'hindi',
    label: 'Hindi',
    a: 'मुंबई में आज भारी बारिश हुई। कई इलाकों में पानी भर गया।',
    b: 'आज मुंबई में तेज़ बारिश हुई। कई इलाकों में जलभराव हो गया।',
  },
]

export const LIMITS = [
  ['Computational cost', '≈285 ms per pair; distillation could cut this 5–10×.'],
  ['Interpretability', 'Per-model scores are visible, but not which passages drove a decision.'],
  ['Language bias', 'Underlying transformers are English-centric.'],
  ['Evaluation noise', 'Single run, no significance tests.'],
  ['Robustness', 'Adversarially perturbed text is untested.'],
]
