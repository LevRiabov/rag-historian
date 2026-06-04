/**
 * mini-projects/06-embeddings-playground — does cosine similarity actually work?
 *
 * Module 3 deliverable. 100 sentences across 8 clusters (including
 * intentional adversarial samples) → embed with BGE-M3 → store in pgvector →
 * run 10 probe queries → see where semantic search wins and where it bites.
 *
 * What this teaches:
 *   - At 100 samples the wins are no longer cherry-picked: if "Who killed
 *     Caesar?" still surfaces Brutus first, that's a real signal.
 *   - Limits become visible: keyword traps (Caesar salad), metaphorical use
 *     (Pyrrhic victory in business), homonyms (Hannibal Lecter vs Barca),
 *     synonym gaps (tyrant vs emperor), multilingual queries.
 *   - The TOP-1 hit being right is easy; getting a clean top-5 with NO
 *     contamination is harder and matters more for RAG.
 *
 * Run:
 *   pnpm dev mini-projects/06-embeddings-playground/index.ts
 *
 * Requirements:
 *   - Postgres + pgvector running:        docker compose up -d
 *   - llama-swap on :8080 serving the `bge-m3` profile (C:\llm)
 *
 * Idempotent: clears `chunks` rows where source='playground' at start.
 */
import 'dotenv/config';
import pg from 'pg';
import pgvector from 'pgvector/pg';

import { createEmbedder, type Embedder, formatCost } from '../../lib/index.ts';

// ---------------------------------------------------------------------------
// Sample data — 100 sentences across 8 clusters.
//
// Cluster taxonomy (deliberately coarse — fine taxonomies pretend semantic
// space is partitioned, but it isn't; "republic" and "empire" overlap, and
// that's the point):
//
//   caesar         — Julius Caesar specifically
//   republic       — Roman Republic (not Caesar)
//   empire         — Roman Empire / emperors after the Republic fell
//   military       — Roman wars, battles, legions, tactics
//   ancient_other  — Greek, Egyptian, Persian — near-misses for "ancient" queries
//   non_ancient    — medieval & modern history (temporal discrimination test)
//   modern_life    — everyday modern stuff (pure control group)
//   trap           — keyword/homonym/metaphor traps that LOOK Roman but aren't
//
// Traps are the headline feature. If similarity is purely keyword-driven,
// the traps will surface for the wrong queries. If BGE-M3 is doing real
// semantic work, the traps stay confined to queries that genuinely match
// their meaning (food, finance, movies, etc.).
// ---------------------------------------------------------------------------
type Cluster =
  | 'caesar'
  | 'republic'
  | 'empire'
  | 'military'
  | 'ancient_other'
  | 'non_ancient'
  | 'modern_life'
  | 'trap';

interface Sample {
  text: string;
  cluster: Cluster;
}

const samples: Sample[] = [
  // === caesar (12) ===
  {
    cluster: 'caesar',
    text: 'Julius Caesar crossed the Rubicon in 49 BC, igniting civil war against Pompey.',
  },
  {
    cluster: 'caesar',
    text: 'Brutus and a group of senators assassinated Caesar on the Ides of March, 44 BC.',
  },
  {
    cluster: 'caesar',
    text: 'Caesar defeated Pompey decisively at the Battle of Pharsalus in 48 BC.',
  },
  {
    cluster: 'caesar',
    text: 'After Caesar fell, his great-nephew Octavian inherited his political legacy.',
  },
  {
    cluster: 'caesar',
    text: 'Caesar wrote commentaries on the Gallic War describing his conquest of Gaul.',
  },
  {
    cluster: 'caesar',
    text: 'The Senate granted Caesar the title dictator perpetuo shortly before his death.',
  },
  { cluster: 'caesar', text: 'Caesar reformed the Roman calendar, introducing the Julian system.' },
  {
    cluster: 'caesar',
    text: 'Mark Antony delivered the famous funeral oration after Caesar was killed.',
  },
  {
    cluster: 'caesar',
    text: 'Cleopatra bore Caesar a son named Caesarion during his time in Egypt.',
  },
  {
    cluster: 'caesar',
    text: 'Caesar pardoned many former enemies after winning the civil war, a policy called clementia.',
  },
  {
    cluster: 'caesar',
    text: 'Caesar\'s final words to Brutus, according to Suetonius, may have been "Et tu, Brute?"',
  },
  {
    cluster: 'caesar',
    text: 'The Battle of Alesia saw Caesar besiege Vercingetorix and end Gallic resistance.',
  },

  // === republic (12) ===
  {
    cluster: 'republic',
    text: 'The Roman Senate held primary legislative power throughout the Republic.',
  },
  {
    cluster: 'republic',
    text: 'Two consuls were elected annually and held imperium over the army and state.',
  },
  {
    cluster: 'republic',
    text: 'The Conflict of the Orders gave plebeians equal political rights with patricians.',
  },
  {
    cluster: 'republic',
    text: 'Cicero served as consul in 63 BC and exposed the Catiline conspiracy.',
  },
  { cluster: 'republic', text: 'The Twelve Tables formed the earliest written code of Roman law.' },
  {
    cluster: 'republic',
    text: 'Tribunes of the plebs could veto actions of magistrates to protect ordinary citizens.',
  },
  {
    cluster: 'republic',
    text: 'The Gracchi brothers tried to redistribute public land to landless veterans.',
  },
  {
    cluster: 'republic',
    text: 'Sulla marched on Rome and was appointed dictator to reform the constitution.',
  },
  {
    cluster: 'republic',
    text: 'The cursus honorum was the sequential ladder of public offices ambitious Romans climbed.',
  },
  {
    cluster: 'republic',
    text: 'Censors counted citizens and assigned them to property classes for voting and taxation.',
  },
  {
    cluster: 'republic',
    text: 'Roman religion was tightly woven into public life through state-appointed priests.',
  },
  {
    cluster: 'republic',
    text: 'The First Triumvirate informally united Pompey, Crassus, and Caesar against Senate opposition.',
  },

  // === empire (12) ===
  {
    cluster: 'empire',
    text: 'Augustus became the first Roman emperor after defeating Antony and Cleopatra at Actium.',
  },
  {
    cluster: 'empire',
    text: 'Tiberius retreated to Capri and let Sejanus run the empire in his absence.',
  },
  {
    cluster: 'empire',
    text: 'Caligula reportedly made his horse Incitatus a senator, illustrating his erratic reign.',
  },
  { cluster: 'empire', text: 'Claudius oversaw the Roman invasion of Britain in 43 AD.' },
  {
    cluster: 'empire',
    text: 'Nero was blamed for the Great Fire of Rome in 64 AD and persecuted early Christians.',
  },
  {
    cluster: 'empire',
    text: "The Year of the Four Emperors followed Nero's death and ended with Vespasian on the throne.",
  },
  {
    cluster: 'empire',
    text: 'Marcus Aurelius wrote the Meditations while commanding legions on the Danube frontier.',
  },
  {
    cluster: 'empire',
    text: 'Diocletian split the empire into the Tetrarchy to make it governable.',
  },
  {
    cluster: 'empire',
    text: 'Constantine legalized Christianity through the Edict of Milan in 313 AD.',
  },
  {
    cluster: 'empire',
    text: 'The Praetorian Guard often decided imperial successions through coup or assassination.',
  },
  {
    cluster: 'empire',
    text: 'The Western Roman Empire fell in 476 AD when Odoacer deposed Romulus Augustulus.',
  },
  {
    cluster: 'empire',
    text: "The emperor Hadrian built a wall across northern Britain to mark the empire's edge.",
  },

  // === military (12) ===
  {
    cluster: 'military',
    text: 'The Roman legion was organized into cohorts and centuries with strict discipline.',
  },
  {
    cluster: 'military',
    text: 'Legionaries built fortified camps every night during a campaign, regardless of distance.',
  },
  {
    cluster: 'military',
    text: 'The pilum was a heavy javelin designed to bend on impact and disable enemy shields.',
  },
  {
    cluster: 'military',
    text: 'Hannibal led Carthaginian forces across the Alps during the Second Punic War.',
  },
  {
    cluster: 'military',
    text: 'Scipio Africanus defeated Hannibal at the Battle of Zama in 202 BC.',
  },
  {
    cluster: 'military',
    text: 'The Battle of Cannae was one of the worst defeats in Roman military history.',
  },
  {
    cluster: 'military',
    text: 'Roman siege engines included the testudo formation, battering rams, and siege towers.',
  },
  {
    cluster: 'military',
    text: "Trajan's legions conquered Dacia, bringing enormous gold reserves into the empire.",
  },
  {
    cluster: 'military',
    text: 'The Teutoburg Forest ambush destroyed three legions under Varus in 9 AD.',
  },
  {
    cluster: 'military',
    text: 'Marius reformed the legions by accepting landless recruits and standardizing equipment.',
  },
  {
    cluster: 'military',
    text: 'Roman cavalry was historically weak; auxiliaries filled the gap on the wings.',
  },
  {
    cluster: 'military',
    text: 'The Roman navy crushed Carthaginian fleets using the corvus boarding bridge.',
  },

  // === ancient_other (15) — Greek, Egyptian, Persian — near-misses for "ancient" ===
  {
    cluster: 'ancient_other',
    text: 'Socrates was sentenced to death by an Athenian jury for corrupting the youth.',
  },
  {
    cluster: 'ancient_other',
    text: 'Plato founded the Academy in Athens, the first institution of higher learning in the West.',
  },
  {
    cluster: 'ancient_other',
    text: 'Alexander the Great conquered the Persian Empire by the age of thirty.',
  },
  {
    cluster: 'ancient_other',
    text: 'The Athenian democracy let male citizens vote directly on policy in the assembly.',
  },
  {
    cluster: 'ancient_other',
    text: 'Spartan warriors trained from childhood in the agoge, an austere military education.',
  },
  {
    cluster: 'ancient_other',
    text: 'The Battle of Thermopylae saw three hundred Spartans hold off a vast Persian army.',
  },
  {
    cluster: 'ancient_other',
    text: 'The Library of Alexandria collected scrolls from across the Mediterranean world.',
  },
  {
    cluster: 'ancient_other',
    text: 'Cleopatra was the last active pharaoh of the Ptolemaic dynasty of Egypt.',
  },
  {
    cluster: 'ancient_other',
    text: 'The pyramids at Giza were built as tombs for the Old Kingdom pharaohs.',
  },
  {
    cluster: 'ancient_other',
    text: 'Hieroglyphs were deciphered using the Rosetta Stone in the early 19th century.',
  },
  {
    cluster: 'ancient_other',
    text: 'Cyrus the Great founded the Achaemenid Persian Empire in the sixth century BC.',
  },
  {
    cluster: 'ancient_other',
    text: 'Darius and Xerxes led massive Persian invasions of Greece that ultimately failed.',
  },
  {
    cluster: 'ancient_other',
    text: 'Aristotle tutored the young Alexander before his conquests began.',
  },
  {
    cluster: 'ancient_other',
    text: 'The Iliad and Odyssey are attributed to the Greek poet Homer.',
  },
  {
    cluster: 'ancient_other',
    text: 'Pythagoras founded a religious-philosophical school focused on mathematics and number mysticism.',
  },

  // === non_ancient (10) — medieval + modern history to test temporal discrimination ===
  {
    cluster: 'non_ancient',
    text: 'Charlemagne was crowned Emperor of the Romans by the Pope on Christmas Day, 800 AD.',
  },
  {
    cluster: 'non_ancient',
    text: 'The Magna Carta limited the powers of the English king in 1215.',
  },
  {
    cluster: 'non_ancient',
    text: "The Black Death killed perhaps a third of Europe's population in the 1340s.",
  },
  {
    cluster: 'non_ancient',
    text: 'Napoleon crowned himself Emperor of the French in Notre-Dame in 1804.',
  },
  {
    cluster: 'non_ancient',
    text: 'The American Revolutionary War ended with the Treaty of Paris in 1783.',
  },
  {
    cluster: 'non_ancient',
    text: 'World War I began after the assassination of Archduke Franz Ferdinand in Sarajevo.',
  },
  {
    cluster: 'non_ancient',
    text: 'The Berlin Wall fell in November 1989, signaling the end of the Cold War in Europe.',
  },
  {
    cluster: 'non_ancient',
    text: 'The French Revolution overthrew the monarchy and established a republic in the 1790s.',
  },
  {
    cluster: 'non_ancient',
    text: "Gutenberg's printing press transformed European literacy in the mid-15th century.",
  },
  {
    cluster: 'non_ancient',
    text: 'The signing of the Declaration of Independence took place in Philadelphia in 1776.',
  },

  // === modern_life (15) — everyday modern, the control group ===
  {
    cluster: 'modern_life',
    text: 'I had a margherita pizza for lunch and the basil was incredibly fresh.',
  },
  {
    cluster: 'modern_life',
    text: 'The weather forecast predicts heavy rain across the city tomorrow afternoon.',
  },
  {
    cluster: 'modern_life',
    text: 'My laptop battery only lasts about four hours these days, it might need replacing.',
  },
  {
    cluster: 'modern_life',
    text: "The new electric cars charge significantly faster than last year's models.",
  },
  {
    cluster: 'modern_life',
    text: 'She brewed a strong espresso and sat down to read the morning paper.',
  },
  {
    cluster: 'modern_life',
    text: 'Streaming services have largely replaced cable television in most American households.',
  },
  {
    cluster: 'modern_life',
    text: 'The marathon runners gathered at the starting line just before sunrise.',
  },
  {
    cluster: 'modern_life',
    text: 'Our flight was delayed three hours due to thunderstorms over the airport.',
  },
  {
    cluster: 'modern_life',
    text: 'A good pair of running shoes can prevent most common joint injuries.',
  },
  {
    cluster: 'modern_life',
    text: "The local farmers market has the best peaches I've had in years.",
  },
  {
    cluster: 'modern_life',
    text: 'JavaScript and Python remain the two most popular programming languages for beginners.',
  },
  {
    cluster: 'modern_life',
    text: 'My dog barks at every delivery driver no matter how many times they visit.',
  },
  {
    cluster: 'modern_life',
    text: 'The new coffee shop downtown has surprisingly good pastries and free Wi-Fi.',
  },
  {
    cluster: 'modern_life',
    text: 'Most modern smartphones can shoot high-quality video without any external equipment.',
  },
  {
    cluster: 'modern_life',
    text: 'I finally finished assembling that bookshelf and only had two screws left over.',
  },

  // === trap (12) — the headline feature ===
  // Keyword traps: words that look Roman but aren't (Caesar salad, Caesarean section)
  {
    cluster: 'trap',
    text: 'The Caesar salad was invented in Tijuana in 1924 by restaurateur Caesar Cardini.',
  },
  {
    cluster: 'trap',
    text: 'A classic Caesar dressing combines anchovies, garlic, parmesan, lemon, and egg yolk.',
  },
  {
    cluster: 'trap',
    text: 'A Caesarean section is a surgical delivery used when vaginal birth would be risky.',
  },
  // Homonym traps: same name, different person/thing
  {
    cluster: 'trap',
    text: 'Hannibal Lecter is a fictional cannibal psychiatrist created by Thomas Harris.',
  },
  {
    cluster: 'trap',
    text: 'Brutus is also the name of a popular brand of work boots sold across Europe.',
  },
  {
    cluster: 'trap',
    text: 'Pompey is a coastal city in southern England, known for its naval dockyards.',
  },
  {
    cluster: 'trap',
    text: 'Trajan is the name of a widely-used typeface designed by Carol Twombly in 1989.',
  },
  // Metaphorical / modern uses of Roman terms
  {
    cluster: 'trap',
    text: 'The startup won the deal but burned so much cash it was a Pyrrhic victory.',
  },
  {
    cluster: 'trap',
    text: 'The journalist crossed her own Rubicon when she published the leaked source code.',
  },
  {
    cluster: 'trap',
    text: "After the merger, the new CEO's first hundred days felt like a corporate triumph.",
  },
  {
    cluster: 'trap',
    text: 'The phrase "Et tu, Brute?" is now used jokingly when a friend turns on you.',
  },
  {
    cluster: 'trap',
    text: 'The Praetorian was a 2008 video game about a Roman soldier (in name only — actually a shooter).',
  },
];

// ---------------------------------------------------------------------------
// Probe queries — 10 questions, mix of clean wins and intentional stress tests.
//
// `expectedClusters` lists what SHOULD show up in the top results. Empty
// array means exploratory (no clean expected behavior — we're just curious).
// `adversarial: true` flags probes where keyword overlap or metaphor sets
// a trap; the verdict line tells us if BGE-M3 saw through it.
// ---------------------------------------------------------------------------
interface Probe {
  query: string;
  expectedClusters: Cluster[];
  adversarial?: boolean;
  note?: string;
}

const probes: Probe[] = [
  {
    query: 'Who assassinated Julius Caesar?',
    expectedClusters: ['caesar'],
    note: 'Baseline win — should be a clean caesar sweep with Brutus at top.',
  },
  {
    query: 'How did the Roman Republic make laws?',
    expectedClusters: ['republic'],
    note: 'Tests whether republic and empire stay separated.',
  },
  {
    query: 'Which Roman emperor was the most cruel?',
    expectedClusters: ['empire'],
    note: "Should pull Nero, Caligula. Watch for caesar leakage — Caesar wasn't an emperor.",
  },
  {
    query: 'Famous battles and military tactics of antiquity',
    expectedClusters: ['military', 'ancient_other'],
    note: 'Ancient_other should ALSO match (Thermopylae) — multi-cluster expected.',
  },
  {
    query: 'How do you make a Caesar salad?',
    expectedClusters: ['trap'],
    adversarial: true,
    note: 'Keyword trap. PURE semantic search wins → trap cluster ONLY. Keyword bias → caesar contamination.',
  },
  {
    query: 'A Pyrrhic victory in modern business',
    expectedClusters: ['trap'],
    adversarial: true,
    note: 'Metaphorical use should beat literal historical use. BGE-M3 vs surface-level matching.',
  },
  {
    query: 'Ancient Greek philosophers',
    expectedClusters: ['ancient_other'],
    note: 'Temporal + geographic discrimination — should NOT pull Roman.',
  },
  {
    query: 'Hannibal eating people',
    expectedClusters: ['trap'],
    adversarial: true,
    note: 'Homonym test. Lecter (cannibal) vs Barca (general). The "eating" should disambiguate.',
  },
  {
    query: 'What is tyranny in the ancient world?',
    expectedClusters: ['empire', 'republic', 'ancient_other'],
    note: 'Synonym mismatch — "tyranny" never appears in our samples. Tests vocabulary bridging.',
  },
  {
    query: 'Кто убил Юлия Цезаря?',
    expectedClusters: ['caesar'],
    note: 'Multilingual probe (Russian: "Who killed Julius Caesar?"). BGE-M3 is trained on 100+ langs.',
  },
];

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------
const embedder: Embedder = createEmbedder({ provider: 'llamacpp' });

const databaseUrl = process.env.DATABASE_URL;
if (!databaseUrl) {
  throw new Error('DATABASE_URL is not set. Check .env and `docker compose up -d`.');
}
const db = new pg.Client({ connectionString: databaseUrl });
await db.connect();
await pgvector.registerType(db);

// ---------------------------------------------------------------------------
// Step 1 — wipe prior playground rows so the script is idempotent.
// ---------------------------------------------------------------------------
await db.query("DELETE FROM chunks WHERE source = 'playground'");
console.log(`Embedder: ${embedder.label}`);
console.log(`Wiped prior playground rows. Embedding ${samples.length} sentences...\n`);

// ---------------------------------------------------------------------------
// Step 2 — embed all samples. With batchSize=32, 100 samples = 4 batched
// HTTP calls (sequential). Output order matches input order.
// ---------------------------------------------------------------------------
const embedResult = await embedder.embed(samples.map((s) => s.text));

console.log(`Embedded ${samples.length} samples in ${embedResult.latencyMs}ms`);
console.log(`Tokens: ${embedResult.usage.inputTokens} (LM Studio often reports 0)`);
console.log(`Cost:   ${formatCost(embedResult.cost)}`);
console.log(`Vector dimension: ${embedder.dimension}\n`);

if (embedder.dimension !== 1024) {
  throw new Error(`Expected 1024-dim vectors for embedding_bge, got ${embedder.dimension}.`);
}

// ---------------------------------------------------------------------------
// Step 3 — bulk insert. Single multi-row INSERT is dramatically faster than
// 100 round-trips. Builds `($1,$2,$3,$4), ($5,$6,$7,$8), ...` parameterized
// SQL — still safe from injection, just fewer round-trips.
// ---------------------------------------------------------------------------
const values: unknown[] = [];
const tuples: string[] = [];
for (let i = 0; i < samples.length; i++) {
  const sample = samples[i];
  const vector = embedResult.vectors[i];
  if (!sample || !vector) continue;
  const base = i * 4;
  tuples.push(`($${base + 1}, $${base + 2}, $${base + 3}, $${base + 4})`);
  values.push(sample.text, pgvector.toSql(vector), 'playground', { cluster: sample.cluster });
}
const tInsert = Date.now();
await db.query(
  `INSERT INTO chunks (text, embedding_bge, source, metadata) VALUES ${tuples.join(', ')}`,
  values,
);
console.log(`Inserted ${samples.length} rows in ${Date.now() - tInsert}ms.\n`);

// ---------------------------------------------------------------------------
// Step 4 — probes. For each: print the top 5 with similarity scores, then a
// verdict line: did the expected cluster(s) dominate? How much contamination?
// ---------------------------------------------------------------------------
console.log('=== Nearest-neighbour probes ===\n');

let probesPassed = 0;

for (const probe of probes) {
  const queryVec = await embedder.embedOne(probe.query);

  const result = await db.query<{
    id: number;
    text: string;
    cluster: Cluster;
    similarity: number;
  }>(
    `SELECT
       id,
       text,
       metadata->>'cluster' AS cluster,
       1 - (embedding_bge <=> $1) AS similarity
     FROM chunks
     WHERE source = 'playground'
     ORDER BY embedding_bge <=> $1
     LIMIT 5`,
    [pgvector.toSql(queryVec)],
  );

  const header = probe.adversarial ? '[ADVERSARIAL]' : '             ';
  console.log(`${header} Q: "${probe.query}"`);
  console.log(`              Expected: ${probe.expectedClusters.join(', ') || '(exploratory)'}`);
  if (probe.note) console.log(`              Note: ${probe.note}`);

  for (const row of result.rows) {
    const isExpected = probe.expectedClusters.includes(row.cluster);
    const marker = isExpected ? '✓' : '✗';
    const sim = Number(row.similarity).toFixed(3);
    console.log(`  ${marker} [${sim}] (${row.cluster.padEnd(13)}) ${row.text}`);
  }

  // Verdict — counts how many of top-5 matched expected clusters.
  const topMatches = result.rows.filter((r) => probe.expectedClusters.includes(r.cluster)).length;
  const top1Match = result.rows[0] && probe.expectedClusters.includes(result.rows[0].cluster);
  const verdict =
    probe.expectedClusters.length === 0
      ? 'EXPLORATORY (no fixed expectation)'
      : top1Match && topMatches >= 4
        ? 'CLEAN WIN'
        : top1Match
          ? `PARTIAL (top-1 ok, ${topMatches}/5 in expected clusters)`
          : `MISS (top-1 is ${result.rows[0]?.cluster}, ${topMatches}/5 in expected)`;
  console.log(`  → Verdict: ${verdict}\n`);

  if (top1Match) probesPassed++;
}

console.log(`=== Summary: ${probesPassed}/${probes.length} probes hit top-1 match ===`);

await db.end();
