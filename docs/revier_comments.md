Paper ID
59
Paper Title
A Filtering-Theoretic Interpretation of Transformers A State-Space Perspective
Reviewer #1
Questions
1. Significance to IVMSP – Potential importance and relevance of the work to the IVMSP community.
4=High significance: The work addresses an important problem and is likely to interest many researchers in IVMSP.
2. Novelty – How novel are the contributions of this paper (e.g., new methods, findings, problem formulations, datasets, or insights)?
4=High novelty: Introduces clearly new ideas, methods, or perspectives.
3. Experimental Study – Are the experimental results sufficient to support the claims of the paper?
4=Strong: Experiments are well designed and provide convincing evidence.
4. Technical Correctness – Are the proposed methods sound and technically correct?
4=Strong: The approach is technically sound and well justified.
5. Clarity of Presentation – Is the paper clearly written and well organized?
4=Good: The paper is clearly written and well organized.
6. Confidence in Your Review – How confident are you in your evaluation?
4=High confidence: I checked the main ideas and experiments carefully.
7. Comments to Authors
This paper presents a theoretical framework that interprets Transformers from a statistical signal processing perspective, establishing a connection between self-attention and adaptive filtering, particularly Wiener filtering. By introducing a learned state-space representation, the authors reformulate attention as a form of non-parametric estimation, where historical samples are adaptively weighted based on similarity in the latent space. However, the work is primarily theoretical and lacks sufficient empirical support. The experimental validation is limited to acoustic echo cancellation, making it unclear whether the proposed interpretation generalizes to broader domains such as vision or NLP. Additionally, the paper does not provide quantitative evidence that this theoretical perspective leads to improvements in performance, efficiency, or interpretability. Furthermore, certain derivations rely on approximations that are not strictly validated, which slightly weakens the overall theoretical rigor.
Weakness
1、Experimental evaluation is limited to acoustic echo cancellation, raising concerns about the generalizability of the proposed framework to other domains such as vision or NLP.
2、The paper does not provide quantitative comparisons demonstrating that the proposed interpretation leads to improved performance, efficiency, or interpretability in practical applications.
3、The role of multi-head attention is only briefly discussed and lacks a concrete theoretical or experimental analysis within the proposed framework.
4、The learned state space interpretation is supported mainly by qualitative visualization (e.g., PCA), without rigorous quantitative metrics to validate structural consistency.
5、The derivation from Gaussian kernel to dot-product attention involves approximations that are not strictly validated, potentially weakening the theoretical rigor.
Soundness
1、The paper provides a novel and insightful theoretical perspective that bridges Transformer architectures with classical adaptive filtering theory.
2、The derivation of attention weights from the Maximum Entropy principle offers an elegant and principled explanation of the Softmax mechanism.
3、The proposed state-space viewpoint presents a coherent framework for understanding representation learning in non-stationary environments.
4、The interpretation of self-attention as a form of non-parametric statistical estimation is conceptually compelling and well-motivated.
5、The paper is clearly structured and systematically develops the connection from Wiener filtering to modern attention mechanisms.
Reviewer #2
Questions
1. Significance to IVMSP – Potential importance and relevance of the work to the IVMSP community.
3=Moderate significance: The work is relevant and may be useful to a portion of the IVMSP community.
2. Novelty – How novel are the contributions of this paper (e.g., new methods, findings, problem formulations, datasets, or insights)?
3=Moderate novelty: Some new elements or combinations of existing ideas.
3. Experimental Study – Are the experimental results sufficient to support the claims of the paper?
2=Weak: Experimental evaluation is limited or lacks important comparisons.
4. Technical Correctness – Are the proposed methods sound and technically correct?
3=Moderate: The methods appear generally sound but some details are unclear.
5. Clarity of Presentation – Is the paper clearly written and well organized?
3=Fair: The paper is understandable but could be better structured or explained.
6. Confidence in Your Review – How confident are you in your evaluation?
3=Moderate confidence: I understand the topic but may have missed some details.
7. Comments to Authors
The paper presents an interesting signal-processing perspective on Transformers by interpreting self-attention through adaptive filtering, local statistical estimation, and state-space reasoning. I appreciate the attempt to connect Transformer attention with Wiener filtering and non-parametric estimation, which is relevant to the IVMSP community and intellectually interesting.

That said, I have several concerns that prevent me from recommending acceptance in its current form.

First, the theoretical claims appear stronger than what is actually established. The derivation from distance-based similarity to scaled dot-product attention relies on multiple approximations and assumptions, including normalized or smoothly varying states and inner-product dominance. Likewise, Theorem 1 is presented only with a proof sketch, while the paper repeatedly uses strong formulations such as “mathematically recovered” and “proved.” In my view, the work would be stronger if it were positioned more explicitly as an interpretation under restrictive assumptions, rather than as a general theoretical equivalence.

Second, the experimental section is too limited to support such broad claims. The empirical validation appears restricted to a synthetic AEC/spatial-tracking setting with 15 trajectories, qualitative attention-map observations, and PCA-based geometry comparisons. While these results are useful as an illustration, they do not convincingly validate a broad filtering-theoretic interpretation of Transformers. I would expect more quantitative evidence, such as ablations on the assumptions, comparisons with alternative similarity kernels or estimation schemes.

Third, the connection to actual Transformer architectures remains incomplete. The discussion focuses on a simplified single-head setting, whereas practical Transformer blocks include multi-head projections, learned Q/K/V mappings, normalization, residual connections, and feed-forward layers. These gaps should be discussed more explicitly, especially if the paper aims to make claims about Transformers as used in practice.

Finally, the manuscript does not appear to follow the standard conference formatting/template closely, including typography and overall presentation style. While this is not the main scientific issue, it contributes to the impression that the submission was not prepared with sufficient care.

Overall, I find the idea promising, but the current submission does not yet provide sufficiently rigorous theory or sufficiently strong experiments for acceptance.
Reviewer #3
Questions
1. Significance to IVMSP – Potential importance and relevance of the work to the IVMSP community.
3=Moderate significance: The work is relevant and may be useful to a portion of the IVMSP community.
2. Novelty – How novel are the contributions of this paper (e.g., new methods, findings, problem formulations, datasets, or insights)?
3=Moderate novelty: Some new elements or combinations of existing ideas.
3. Experimental Study – Are the experimental results sufficient to support the claims of the paper?
2=Weak: Experimental evaluation is limited or lacks important comparisons.
4. Technical Correctness – Are the proposed methods sound and technically correct?
3=Moderate: The methods appear generally sound but some details are unclear.
5. Clarity of Presentation – Is the paper clearly written and well organized?
3=Fair: The paper is understandable but could be better structured or explained.
6. Confidence in Your Review – How confident are you in your evaluation?
3=Moderate confidence: I understand the topic but may have missed some details.
7. Comments to Authors
The paper proposes a theoretical framework for interpreting the Transformer architecture from the perspective of statistical signal processing, aiming to bridge deep learning architectures and classical adaptive filtering theory. Overall, the proposed interpretation is relatively rigorous and well developed. However, the paper is primarily centered on theoretical analysis. On the experimental side, the authors may consider including comparisons with existing Transformer-based approaches as well as traditional methods, so as to better demonstrate the advantages of Transformers in the relevant signal processing settings.

Meta-Review Questions
1. Comments to Authors
The paper proposes a theoretical framework for interpreting the Transformer architecture from the perspective of statistical signal processing.
The theoretical analysis and messages are rigorous and informative.
In the final version, please address the formatting issue.