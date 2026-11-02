# How the gantry figures out who passed and how much to charge

*Em português: [pt-BR/como_funciona.md](pt-BR/como_funciona.md)*

The gantry is the steel frame that spans the highway. Hanging from it, above each lane, are a few
devices. Each one answers a different question: who is this car (tag or plate), what kind of
vehicle is it (how many axles), and how do we prove it passed (photos and time). All of it with
the car at 80 to 120 km/h, with no barrier and no braking.

```
             ┌─────────────────────────────── GANTRY ───────────────────────────────┐
             │  [antenna]   [plate camera + IR flash]   [laser]   [wide camera]      │   one set per lane
             └───────────────────────────────────────────────────────────────────────┘
                    │                 │                 │
  0 to 16 m before ─┼── antenna zone ─┼─ plate photo ───┼──── under the gantry ──→ direction of traffic
                    │                 │  (10 m before)  │
                    ▼                 ▼                 ▼
              reads the tag     photographs plate   measures the vehicle, counts axles
                    └────────┬────────┴────────┬────────┘
                             ▼                 ▼
                    computer on the gantry combines the three reads
                             ▼
                    evidence (~1 MB) + billing record (~2 KB) → concessionaire's system
                             ▼
                 with a tag: the charge goes to the tag company (Sem Parar, ConectCar, Veloe...)
                 no tag: charged by plate, 30 days to pay; not paying is a traffic offense (art. 209-A of Brazil's traffic code)
```

## The tag

The tag is the sticker on the windshield. Inside it there is a chip and a printed antenna, and no
battery.

The gantry's antenna sends out radio waves (in the 902 to 928 MHz band, the same used by
anti-theft tags in stores). The wave itself powers the chip, which "wakes up" and answers by
reflecting the signal back with its code. The technical standard is ISO/IEC 18000-63, the same
one used in logistics.

The read takes a few thousandths of a second and repeats several times while the car crosses the
antenna's zone. The system also records the signal strength, which helps tell which lane the car
was in.

Any authorized tag works at any federal highway gantry, thanks to an ANTT rule from 2014
(Resolution 4,281). Tag users also get a discount.

## The plate

Reading plates with a camera has a technical name, ANPR (automatic number plate recognition), and
it is the same kind of problem as the fish-tracking video: find the object in the image, follow it
as it moves, and pull a piece of information out of it.

1. The photo. The cameras are black and white and have an infrared flash that fires together with
   the shutter. The exposure lasts millionths of a second, enough to freeze a car at 120 km/h.
   Infrared works day and night, isn't thrown off by headlights or the sun in front, and makes the
   plate's reflective film glow. There is usually one camera for the front plate and one for the
   rear plate in each lane.
2. The right moment. The video itself, a laser or a radar notices the car coming and picks the best
   frames when it crosses an imaginary line, about 10 meters before the gantry.
3. Finding the plate. A computer-vision program (YOLO-style) draws a box around the plate.
4. Straightening. The crop is corrected so it isn't tilted or too dark.
5. Reading. Another program reads the seven letters and digits. The Mercosur format helps check:
   in `ABC1D23` the third position is always a letter, so a zero there can only be the letter O.
6. Deciding. Each read comes with a confidence level. The reads from several photos, front and
   rear, are combined. If the confidence stays low, a person looks at the photo.

The most common swaps are O and 0, B and 8, I and 1, S and 5, Z and 2. Dirty, bent or covered
plates, or plates hidden by a trailer, also get in the way. When the car has a tag, the plate
registered with the tag is used as a check. If the tag says `ABC1D23` and the camera read
`ABC1023`, the tag wins. If the two have nothing to do with each other, it may be a tag stuck on
the wrong car or a cloned plate, and the case goes to investigation.

## The type of vehicle

A laser above each lane sweeps the vehicle and builds a 3D outline of it: length, width, height
and number of axles. That gives the toll class. A passenger car pays one rate, trucks and buses
pay per axle, and a raised axle doesn't count.

The class registered with the tag is compared with the one the laser measured. That is how the
system catches, for example, a truck using a car's tag.

## The decision

The gantry's computer combines the three reads by time and lane:

| What happened | What the system does |
|---|---|
| The tag was read and is valid | bills the tag and uses the plate as a check |
| No tag, plate read with confidence | bills by plate |
| No tag, doubtful read | a person checks the photos |
| Tag and plate don't match | investigation (may be fraud) |

The result of each pass is a small record, about 2 KB, with gantry, lane, date, time, plate or
tag, class and confidence level. With it goes an evidence package of about 1 MB, with the photos,
kept in case someone disputes the charge. How much that adds up to per minute is worked out in
[`src/data_volume.py`](../src/data_volume.py) and in the README.

## Where data science comes in

Millions of passes a month flow from the gantries to the cloud. Someone has to merge repeated
reads of the same car and organize everything by gantry and by date.

You can measure how often the camera is right, which characters it confuses most, and how much
rain and darkness hurt the read.

You can predict who won't pay on time and reach out first, find fraud (tag, plate and class that
don't match; the same plate at two distant gantries within an impossible time), and forecast the
traffic at each gantry by hour and season.

Three of these ideas are tested in [`src/ml/`](../src/ml/) and explained in the README.
