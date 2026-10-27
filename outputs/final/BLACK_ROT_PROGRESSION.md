# Black rot (Xanthomonas campestris pv. campestris): progression, weather risk, actions

Companion to the downy mildew page in `PROJECT-1 MAIN DOC.pdf`, same layout.
Status: draft from extension and research sources (listed at the end). Not expert-verified.

## Progression of symptoms

### On mature leaves (the usual order)

1. **Dull yellow patches at the leaf edge.** Bacteria enter through the water pores (hydathodes) on the leaf margin, so the first sign is irregular, dull yellow blotches on the edge, not in the middle of the leaf. At about 25 °C they appear 10-14 days after infection, and later in cooler weather. (Wisconsin, AHDB, AGES)
2. **The patch becomes a yellow V (wedge).** The wide end sits on the leaf edge and the point aims at the midrib or leaf base. (Wisconsin, Kentucky, UConn, PNW)
3. **Veins inside the V turn brown, then black.** This is the key sign. It is easiest to see when the leaf is held up to the light. (Wisconsin, AHDB, Kentucky)
4. **The centre of the V turns brown, dry and papery, with a yellow border.** The blocked veins cut off water, so the tissue dies dry rather than rotting. (Wisconsin, AHDB, UConn)
5. **Whole leaves yellow, wilt and drop early, lower leaves first.** On cauliflower, losing the wrapper leaves exposes the curd, which makes it unmarketable. (Kentucky, Illinois, AHDB)
6. **Systemic symptoms: black veins in the stem, one-sided wilt and stunting.** A cross-cut of the petiole or stem shows a black ring. Plants often wilt or stunt on one side only, and roots can blacken. V-lesions may then appear anywhere on the plant, not only at the edges. (AHDB, PNW, UConn, Illinois, Wisconsin)
7. **Head damage.** Heads stay small and of poor quality. Cutting the head lengthwise shows blackened veins. Soft-rot bacteria often invade next, causing slimy, foul-smelling decay of the stalk or head. (KAU, Kentucky)

Unlike downy mildew, black rot never produces fuzzy growth, so weather does not switch a visible sign on or off. Infection through insect-feeding holes or wounds gives lesions away from the edge. (Kentucky)

### On seedlings (different order)

1. **The cotyledon (seed-leaf) margins turn black first.** (Illinois, UConn, KAU)
2. **The cotyledons shrivel and drop off, often before anyone notices.** The bacteria stay in the plant at low levels, and the infection usually has already reached the stem. (AHDB, PNW, Kentucky)
3. **Yellow V-lesions with black veins on the first true leaves.** (UConn)
4. **The seedling yellows, stunts, drops its lower leaves, wilts and collapses.** (Illinois, KAU, Kentucky)

Transplants can carry the infection with no visible symptoms. Symptoms show 1-2 weeks after infection, so healthy-looking seedlings can still bring the disease into the field. (UConn, AHDB)

## Table 1: Weather risk (from sensors)

| Weather risk | Temperature | Relative humidity | Wetness / rain (last 3 days) | Dew point depression |
|---|---|---|---|---|
| **Low** | Below 18 °C or above 35 °C | Below 80% | No rain or overhead irrigation; leaves dry by early morning | Above 3 °C |
| **Medium** | 18-24 °C or 31-35 °C | 80-85% | Dew or fog nights, light rain | 1.5-3 °C |
| **High** | 25-30 °C (up to 33 °C in Indian plains) | Above 85% | Rain, wind-driven rain or storms, sprinkler irrigation | Under 1.5 °C |

Where the numbers come from:
- **Temperature:** the optimum is 25-30 °C (UIUC 77-86 °F, PNW 80-86 °F, UConn, Pathogens review). Kentucky gives 79-95 °F. In Udaipur, cauliflower black rot was worst at 30-33 °C with RH above 82-89%. The bacteria grow slowly from 5 to 35 °C, and infected hosts are reported to show no symptoms below 18 °C (Pathogens 2026 review). That review is secondary, and I could not open the full text.
- **Wetness:** free water from dew, fog, rain or sprinklers is needed for infection and spread (UIUC, PNW). Most local spread is by rain splash and wind-driven rain (AHDB). Early-morning guttation drops at the leaf edge are a natural entry route (Pathogens review). In Meghalaya, black rot incidence correlated with rainfall (r = 0.88), temperature (0.91) and humidity (0.87), and peaked in the monsoon (ICAR IJAgS).
- **My design choices, not published thresholds:** the 80/85% RH cut-offs (anchored on the Udaipur 82-89% figure) and the dew-point-depression column (a proxy for dew/guttation copied from the downy mildew table). Unlike downy mildew, I found **no published leaf-wetness-hours threshold** for black rot. So the "wetness" column uses rain and irrigation events instead of "consecutive hours RH > 90%".
- **Time lag:** symptoms appear 10-14 days after infection (AHDB, AGES). When the camera sees nothing, take the weather risk as the worst level over roughly the last 14 days, not just today.

## Table 2: Weather risk x camera confidence -> action

| Weather risk | Camera: none | Camera: medium | Camera: high |
|---|---|---|---|
| **Low** | No action | Re-scan and verify (cut a leaf: black veins?) | Verify, then remove affected leaves (see below) |
| **Medium** | No action, monitor; check plants after rain | **Remove** affected leaves; stop overhead irrigation | **Remove** affected leaves/plants; no field work while wet |
| **High** | No action, monitor closely; re-scout 10-14 days after storms | **Remove + protect** | **Remove + protect** |

- **Verify:** black or brown veins inside a yellow V that starts at the leaf edge, and no fuzz underneath. A cross-cut petiole shows a black ring.
- **Remove:** take off affected leaves when plants are dry, bag them, and keep them out of the field. Pull the whole plant if the stem veins are black. Clean hands and tools afterwards.
- **Protect:** the neighbouring plants. Ask the local extension officer whether a fixed-copper spray is worth it.

Why "Spray" is not the main action here (unlike downy mildew): black rot is bacterial, and no spray cures infected plants. Kentucky reports that chemical control "has been inconsistent in university trials", and that copper can only slow spread in some cases. The actions that work are cultural: remove infected tissue, keep leaves dry, avoid handling wet plants, use clean seed and transplants, and rotate away from crucifers for 3-4 years.

## Lookalikes to check in the app

- **Downy mildew:** angular yellow patches between the veins, grey-white fuzz underneath, veins not blackened, in cool weather (10-20 °C). Black rot starts at the edge as a V, has black veins, and prefers warm weather.
- **Drought, overwatering or over-fertilisation:** these can also yellow the leaf margins, but the veins do not turn black (Wisconsin).
- **Bacterial soft rot on the head:** often a *secondary* invader of black-rot plants (Kentucky). Check the stem and leaf veins before calling it soft rot only.

## Sources

- Wisconsin Vegetable Pathology, Brassica black rot: https://vegpath.plantpath.wisc.edu/diseases/brassica-black-rot
- PNW Plant Disease Handbook, Cabbage & cauliflower black rot: https://pnwhandbooks.org/plantdisease/host-disease/cabbage-cauliflower-brassica-oleracea-black-rot
- AHDB, Black rot of brassicas symptoms and biology: https://horticulture.ahdb.org.uk/knowledge-library/black-rot-of-brassicas-symptoms-and-biology
- University of Kentucky PPFS-VG-01, Black Rot of Crucifers: https://plantpathology.mgcafe.uky.edu/files/ppfs-vg-01.pdf
- UConn IPM, Black rot of crucifers: https://ipm.cahnr.uconn.edu/black-rot-of-crucifers/
- UConn IPM, Black rot on cole crop transplants: https://ipm.cahnr.uconn.edu/black-rot-on-cole-crop-transplants/
- University of Illinois HortAnswers, Black rot of cole crops: https://web.extension.illinois.edu/hortanswers/detailproblem.cfm?PathogenID=133
- KAU Agri-Infotech, Cauliflower black rot: https://celkau.in/crops/vegetables/cauliflower/blackrot.php
- AGES (Austria), Black rot disease in crucifers: https://www.ages.at/en/plant/plant-health/pests-from-a-to-z/black-rot-disease-in-crucifers
- Pathogens 15(9):974 (2026), Xcc review: https://www.mdpi.com/2076-0817/15/9/974 (temperature and guttation lines seen via search summary only)
- ICAR Indian J. Agric. Sci., Black rot of cabbage in East Khasi Hills, Meghalaya: https://epubs.icar.org.in/index.php/IJAgS/article/view/142761
- MPUAT Udaipur thesis, cauliflower black rot (30-33 °C, RH > 82-89%): https://krishikosh.egranth.ac.in/handle/1/5810032185 (seen via search summary only)
