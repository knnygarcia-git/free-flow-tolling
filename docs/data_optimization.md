# Taking better care of a gantry's data

*Em português: [pt-BR/otimizacao_dados.md](pt-BR/otimizacao_dados.md)*

Here I follow the data in the order it is born: the photo on the gantry, the plate read, the
decision to bill, the checks, storage, and day-to-day monitoring. At each step I say what I would
do and, where I tested it, what happened.

The numbers in parentheses come from the experiments in `src/ml/`, run on simulated data. They
give a sense of the size of the gain, not a promise.

---

## 1. On the gantry

Process the image where it is born. The cameras on one gantry see about 55 GB per minute. Sending
that over the network is out of the question. The computer on the gantry finds the plate, reads
it, decides, and sends only the result: about 2 KB of record and 1 MB of evidence per car.

Keep several photos of each car, but throw away the bad ones before sending. Three to six photos
per camera is what makes the vote in section 2 possible. A photo with a cropped, blurred or
overexposed plate just takes up space.

Get the clocks right. Camera, antenna and laser are matched by time. A tenth of a second, at
100 km/h, is almost 3 meters, enough to pair one car's tag with the photo of the car behind it.
Every device needs to sync its clock over the network and record time in the same time zone.

Record the conditions with each pass: whether it was raining, whether it was dark, the speed, the
state of the equipment. Without that, the alarm in section 6 would have no way of telling a rain
shower from a dirty lens.

## 2. Reading the plate

| Practice | Why | Result in the project |
|---|---|---|
| Mercosur format rule (three letters, a digit, a letter, two digits) | A digit where a letter belongs is a sure error (0 becomes O, 8 becomes B) | 26% fewer misread plates |
| Character-by-character vote across all photos and cameras, weighted by confidence | A random error rarely repeats in every photo | 30% fewer misread plates |
| Separate random errors from repeating ones | A dirty plate is misread the same way every time; more photos don't help | 81% of remaining errors are on bad plates |

## 3. Deciding to bill

Don't treat the camera's confidence as the chance of being right. It tends to be optimistic: a
read with "95% confidence" is right far less often than that (see chart 07). A model that combines
every clue and is tuned against real outcomes gives a probability you can trust.

Pick the cut-off by cost, not because 90% sounds like a good number. With a trustworthy
probability, the math is simple: billing automatically pays off when the chance of being right is
above `1 − cost of a human check ÷ cost of a wrong charge`. With R$ 0.60 and R$ 30, that is 0.98,
and the experiment landed very close to it (0.979).

Use the tag and the plate to watch each other. The clue that weighed most in the model was the
difference between the plate registered with the tag and the photographed plate. That is what
catches a tag stuck on another car, something the rule "tag read, bill it" never sees.

In the test, wrong charges fell from 422 to 17. In exchange, the photos sent to a person rose from
1.2% to 4.3% of passes. All in, the total cost fell 89%.

## 4. Before storing and before billing

Simple automatic checks, run on every pass. When one fails, the reason is logged, so you can later
count which ones fail most.

| Check | What it looks at | If it fails |
|---|---|---|
| Fields | everything arrived, in the right format, from a gantry and lane that exist | reject and alert the technical team |
| Plate format | follows the Mercosur pattern or the old one (three letters and four digits) | try to fix it; if not, send to a person |
| Registry | the plate exists and its registered vehicle class matches what the laser measured | send to a person |
| Tag vs. plate | the tag's plate equals the photographed one (allowing one character off) | fraud investigation |
| Tag vs. laser | the tag's vehicle class equals the measured one | bill by the measured class and investigate |
| Duplicates | the same plate at the same gantry within a few seconds | merge into a single pass |
| Impossible time | the same plate at two gantries at an implied speed above 200 km/h | suspected cloned plate |
| Timestamp | a pass dated in the future, out of order, or with a clock out of sync | correct with server time and alert |

## 5. Storage

Keep the full photo package only when it may become evidence. The full photos (almost 1 MB) matter
when the charge was made by plate, when a person checked it, or when the driver may dispute it. If
the tag was read and the plate photo confirmed it, with a chance of being right above 99.9%, a
slim version is enough: plate crops, laser profile and the record, about 35 KB. Across the 4
simulated gantries, that brings storage down from 33 to 14 GB a day, 59% less. Before doing this
for real, check with the legal team and the concession contract what must be kept and for how
long.

Organize for the people who will query it. Records can live in Parquet files split by date and by
gantry, so a question like "how much did lane 2 read yesterday" opens only the slice it needs.
Photos go to object storage, with a name that encodes gantry, date, time and pass, and the table
stores only that name.

Tier by age. Recent photos, still within the dispute window, stay in fast storage. Then they move
to cheaper, slower storage. At the end of the retention period, they are deleted.

Treat the plate as personal data. A plate linked to time and place reveals someone's routine, and
Brazil's data protection law (LGPD) applies. Limit who can see it, log who looked, and in the
datasets used for analysis and model training, replace the plate with a code that can't be traced
back to it.

## 6. Day to day

Compare each camera with what it should be reading, not with a fixed number. An alarm like "below
85%" goes off every time it rains: 172 false alarms in 3 days. When I compared against the
expected rate for that hour and that weather, the false alarms disappeared, and the two failing
cameras were caught in 10 and 30 minutes.

These are the numbers I would put on a dashboard, per gantry and per lane:

| Number | What it tells you |
|---|---|
| Share billed without anyone checking | whether the process runs on its own |
| Share sent to a person, and the queue size | cost and delay of billing |
| Wrong charges per 10,000 (disputes the driver won) | how often the system is really right |
| Confident reads per camera, against the expected rate | whether the camera and its flash are fine |
| Agreement between front and rear photos | quality of plate reading |
| Tags read, among the cars that have one | whether the antenna is fine |
| Tags that don't match the plate or the laser | fraud |
| Time between the pass and the record reaching the system | health of the network and the gantry |

## 7. Over time

Every photo a person checks is a correct answer you can use for training. Send the cases where
the model is least sure to be checked first. Those teach the most.

There is a trap, though: you only know the truth for the cases someone checked or the driver
disputed. To avoid fooling the model, someone also needs to check a random sample of the charges
made automatically. That is how you measure the errors nobody complained about.

And the world changes. Cameras age, the fleet turns over, new plates appear. I would check every
month whether the model's predicted chance still matches the real outcomes (chart 07) and retrain
when it starts to drift.
