# The residual CCD / Lee / Murphy bias against the Augsburg readers

Backlog task #7. Regenerate every number here with

```bash
python scripts/analyse_reader_bias.py -o out/reader_bias
```

(50 reader-study patients, both legs, ~11 s on 12 processes; measured at `822181b`).

## Conclusion

**The landmarks Lee and Murphy share are no longer biased. The whole residual is
the proximal neck axis, and the two methods err in opposite directions.**

The two methods use the same distal condylar line and the same femoral head centre
and differ only in where they put the other end of the neck axis. So the mean of
their two biases isolates the shared landmarks and half their difference isolates
the proximal axis:

| side | n | shared | method-specific |
| --- | ---: | ---: | ---: |
| left | 47 | **+0.49°** (sd 4.96) | ±4.63° |
| right | 43 | **−0.60°** (sd 3.26) | ±4.29° |

The shared term is zero to within a tenth of its own spread. Directly: the
pipeline's two proximal axes sit **−18.0° / −17.8°** apart (Lee minus Murphy,
sd 4.4 / 5.0) where the readers' sit **−4.3° to −5.8°** apart (sd 4.0–7.7). The
pipeline's gap is ~13° too wide, and it is consistent rather than noisy — its
spread is *narrower* than the readers'. The per-case correlation between the
pipeline's gap and the readers' is only +0.11 to +0.39, so it is not a constant
offset either: the pipeline's Lee axis does not follow the anatomy the readers see.

## The scale: the readers barely agree with each other on CCD

Every bias below has to be read against this.

| measure | side | n | felix − jonathan | mean abs diff | r |
| --- | --- | ---: | ---: | ---: | ---: |
| CCD | left | 32 | −3.45 | 6.51 | **+0.296** |
| CCD | right | 32 | −3.30 | 6.30 | **+0.528** |
| AT (Lee) | left | 49 | −1.19 | 4.06 | +0.875 |
| AT (Lee) | right | 49 | −1.24 | 3.84 | +0.916 |
| AT (Murphy) | left | 49 | +0.09 | 5.66 | +0.774 |
| AT (Murphy) | right | 49 | +0.33 | 4.36 | +0.891 |

## Per measurement

**CCD — not established as a pipeline defect by this data.** The bias is +2.6 to
+3.8° against jonathan and +6.7 to +7.1° against felix, i.e. it is smaller than
the two readers' own mutual disagreement (6.3–6.5° mean absolute, and they differ
by 3.3–3.5° in the mean, in the same direction that separates the two biases). The
pipeline correlates with each reader (+0.40 to +0.58) **better than they correlate
with each other** (+0.30 to +0.53). Worth keeping in view separately: the pipeline
mean is 137.9–139.0° where adult CCD is usually quoted as 125–135° and the readers
sit at 129.9–136.3°, i.e. at the top of that range — so an absolute offset may well
exist, but these readers cannot adjudicate it. A reference with a stated protocol
would be needed.

**Lee — a real −6°, plus a sign convention that is not an error.** The readers
recorded **no negative value in 198 measurements**, exactly the unsigned pattern
already documented for torsion. Splitting on it:

| side | all | pipeline > 0 (no sign ambiguity) | pipeline < 0 |
| --- | ---: | ---: | ---: |
| left | −7.75 (n=50) | **−6.12** (n=39) | −13.54 signed, **+4.11 unsigned** (n=11) |
| right | −8.21 (n=50) | **−5.87** (n=40) | −17.57 signed, **+4.89 unsigned** (n=10) |

On the ~20% of legs the pipeline calls retrotorted, the *magnitude* agrees to
within +4 to +5° and only the sign disagrees — which is a convention mismatch, not
a measurement error. The remaining −6° on the unambiguous legs is well outside the
readers' mutual scatter (they agree to ±4°, r ≈ 0.9), so it is real.

**Murphy — a real +3.6 to +5.3°, at the edge of reader noise.** Consistent in
direction across both readers and both sides, but comparable in size to the
readers' own disagreement on Murphy (4.4–5.7° mean absolute). `CLAUDE.md` already
records that the distal landmark fix did not explain this; the decomposition above
localises what is left to the proximal axis — i.e. the neck-base end, since the
head centre is shared and carries no bias.

## The pipeline is not noisier than the readers

A patient's two femurs share constitutional torsion, so left–right agreement
measures a method's own stability with no reader involved:

| source | CCD | AT (Lee) | AT (Murphy) |
| --- | ---: | ---: | ---: |
| pipeline | **+0.770** (3.60°) | +0.832 (6.86°) | **+0.867** (6.38°) |
| felix | +0.568 (5.45°) | +0.822 (4.83°) | +0.763 (6.45°) |
| jonathan | +0.586 (5.12°) | +0.871 (4.40°) | +0.749 (6.20°) |

(Pearson, with mean |L−R| in brackets.) The pipeline is **more** self-consistent
than either reader on CCD and on Murphy, and comparable on Lee. So the residual is
a definitional offset, not scatter — which is why it shows up as a clean ±4.5°
split rather than as noise.

Note this does *not* corroborate the "Lee is unstable" framing of backlog task #2
on this cohort: Lee's left–right agreement (+0.832) is between the two readers'
(+0.822, +0.871). The ±20–46° jumps that task records are individual failures on
named cases, not general scatter.

## What this leaves open

* The proximal neck axis is the whole remaining story for Lee and Murphy. Lee's is
  the larger error and already has backlog task #2; Murphy's +4.5° has no item yet.
* CCD's absolute level (pipeline ~138° vs a 125–135° literature range) cannot be
  settled against these readers and needs a reference with a stated protocol.
* Only the sign-unambiguous legs can test the torsion sign at all. Asking the
  readers whether they recorded unsigned values would retire that ambiguity for
  this cohort and for the knee-rotation convention (task #16) at the same time.
