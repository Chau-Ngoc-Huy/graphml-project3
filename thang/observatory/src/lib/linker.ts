/** Port of thang/mmxai_eval.py text normalisation + ConceptLinker (greedy longest n-gram match on the label vocabulary). */
const STOPWORDS = new Set(
  `a an the and or but if then than so of at by for with about against between into through during before
after above below to from up down in out on off over under again further once here there when where why how
all any both each few more most other some such no nor not only own same too very s t can will just don
should now is are was were be been being am have has had having do does did doing would could ought i me
my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself
it its itself they them their theirs themselves what which who whom this that these those while as until
one two three four five six seven eight nine ten several many lot lots something someone somebody thing
things another next near around along front top side back also very image picture photo view shot
showing shows show close closeup`.split(/\s+/),
);
const IRREG: Record<string, string> = {
  men: "man", women: "woman", children: "child", people: "people", feet: "foot", teeth: "tooth", mice: "mouse",
  geese: "goose", knives: "knife", leaves: "leaf", shelves: "shelf", wolves: "wolf", glasses: "glasses", jeans: "jeans",
  pants: "pants", shorts: "shorts", scissors: "scissors", bus: "bus", grass: "grass", dress: "dress", skis: "ski",
  fries: "fries", sunglasses: "sunglasses", clothes: "clothes", tennis: "tennis", news: "news", lens: "lens", gas: "gas",
  canvas: "canvas", atlas: "atlas", chess: "chess",
};

export function ruleLemma(w: string) {
  if (w in IRREG || w.endsWith("us") || w.endsWith("is") || w.endsWith("ss")) return IRREG[w] ?? w;
  if (w.length > 4 && w.endsWith("ies")) return w.slice(0, -3) + "y";
  if (w.length > 4 && w.endsWith("es") && "sxzh".includes(w[w.length - 3])) return w.slice(0, -2);
  if (w.length > 3 && w.endsWith("s") && !w.endsWith("ss")) return w.slice(0, -1);
  return w;
}

export interface LinkToken {
  text: string;
  lemma: string;
  concept?: string;
  stop: boolean;
}
export interface LinkResult {
  concepts: string[];
  tokens: LinkToken[];
}

export function linkConcepts(text: string, vocab: Set<string>, maxN = 3): LinkResult {
  const raw = text.toLowerCase().match(/[a-z]+/g) ?? [];
  const lem = raw.map(ruleLemma);
  const tokens: LinkToken[] = raw.map((t, i) => ({ text: t, lemma: lem[i], stop: STOPWORDS.has(lem[i]) }));
  const out: string[] = [];
  let i = 0;
  while (i < lem.length) {
    let hit: [string, number] | null = null;
    for (let n = Math.min(maxN, lem.length - i); n >= 1; n--) {
      const gram = lem.slice(i, i + n);
      if (STOPWORDS.has(gram[0]) || STOPWORDS.has(gram[gram.length - 1])) continue;
      const name = gram.join(" ");
      if (vocab.has(name)) {
        hit = [name, n];
        break;
      }
    }
    if (!hit) {
      i++;
      continue;
    }
    for (let k = i; k < i + hit[1]; k++) tokens[k].concept = hit[0];
    if (!out.includes(hit[0])) out.push(hit[0]);
    i += hit[1];
  }
  return { concepts: out, tokens };
}
