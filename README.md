For detailed documentation, please see here: **https://cbica.github.io/spare_score/**

# Compute SPARE Scores for Your Case
"SPARE" is short for "Spatial Pattern of Abnormalities for Recognition of ..." If you have brain images of a case population, such as the Alzheimer's disease (AD), the SPARE model will try to find characteristic brain patterns of AD with respect to a control population, such as cognitively normal. This would be an example of a classification-based SPARE model (currently powered by support vector machine or SVM). This model (that we named SPARE-AD) then computes SPARE-AD scores on an individual-basis that indicates how much the individual carries the learned brain patterns of AD.

Alternatively, you may want to find the spatial pattern related to brain aging (BA). In this case, you would provide sample images and indicate that chronological age is what you expect the model to learn patterns for. This would be an example of a regression-based SPARE model (also powered by SVM). This model (that we named SPARE-BA) then computes SPARE-BA scores on an individual-basis that predicts your brain age.
<br /><br />

## References
- SPARE-AD

  Davatzikos, C., Xu, F., An, Y., Fan, Y. & Resnick, S. M. Longitudinal progression of Alzheimer's-like patterns of atrophy in normal older adults: the SPARE-AD index. Brain 132, 2026-2035, [doi:10.1093/brain/awp091](https://doi.org/10.1093/brain/awp091) (2009).

- SPARE-BA

  Habes, M. et al. Advanced brain aging: relationship with epidemiologic and genetic risk factors, and overlap with Alzheimer disease atrophy patterns. Transl Psychiatry 6, e775, [doi:10.1038/tp.2016.39](https://doi.org/10.1038/tp.2016.39) (2016).

- diSPARE-AD

  Hwang, G. et al. Disentangling Alzheimer's disease neurodegeneration from typical brain ageing using machine learning. Brain Commun 4, fcac117, [doi:10.1093/braincomms/fcac117](https://doi.org/10.1093/braincomms/fcac117) (2022).

## Disclaimer
- The software has been designed for research purposes only and has neither been reviewed nor approved for clinical use by the Food and Drug Administration (FDA) or by any other federal/state agency.
- By using spare_scores, the user agrees to the following license: https://www.med.upenn.edu/cbica/software-agreement-non-commercial.html

## Contact
For more information and support, please post on the [Discussions](https://github.com/CBICA/spare_score/discussionss) section or contact <a href="mailto:software@cbica.upenn.edu">CBICA Software</a>.
