# Deeper X v2 – changes made and items for you to check

Output: `Deeper_X_v2_IEEE.docx` (built on the IEEE Transactions template: its styles, header, page setup, two-column layout and drop cap).

## Formatting applied from the template
- Title, author line, Abstract (bold, one paragraph), Index Terms, Heading 1/2, body text, figure captions ("Fig. n."), table titles ("TABLE I" above the table), equations (numbers flush right), references and the footnote block all use the template's own styles.
- Index Terms put in alphabetical order (the template requires this).
- Contribution list typed as "1)", "2)" ... so it does not depend on Word list numbering.
- Appendix heading and its title merged into one heading ("Appendix A: Full Hyperparameters").
- Fig. 1 moved to where it is first cited (Section III-B) so the abstract stays in the template's column. Fig. 3 moved one paragraph later so Table X is not split across columns.
- Tables kept together on a page/column; column widths adjusted for Tables IX, X, XI and XV so headings do not wrap.
- Author affiliations moved into the first-page footnote, as IEEE does. Author names, affiliations, dates and funding are still placeholders.

## Wording changes (abbreviations defined at first use, as the template asks)
- Abstract: GFLOPs, GPU, PSNR, SSIM, CE-VAE spelled out.
- Introduction: GFLOPs; LSUI, UIEB, EUVP given full names.
- Related work: CE-VAE, NAFNet, UIQM, UCIQE, PSNR, SSIM.
- III-G: LPIPS, MS-SSIM, FFT. V-E: CPU.
- Table I: added citation [22] to UFO-120 and [5] to UIEB-C60.
No numbers, results or claims were changed.

## Please check (I did not change these – they need your data)
1. LSUI training pairs: Table VIII says 3,679, Section IV-E says 3,685.
2. Joint-run validation images: Section IV-E says 554; 200 + 60 + 300 = 560.
3. Rounding: "gains 7.77 dB" but 25.80 − 18.02 = 7.78; "loses 0.89 dB" but 25.80 − 24.90 = 0.90. Fine if they come from unrounded values.
4. Section V-J (test-time augmentation) does not say which model was tested (19.73 dB on UIEB looks like the LSUI-trained model).
5. Section V-D: UIEB UCIQE "0.6086 is the best" equals the single-run 0.609 in Table XI when rounded.
6. Table XV (ablation) is all "TBD". Reviewers will likely expect results, or the table and Section V-I removed.
7. Table XIV methods (DGC-UWnet, LSNet, USUIR, FUSION, snnTrans-DHZ) and Table I datasets (SUIM-E, SQUID, U45) have no citations.
8. Fill in: authors and IEEE membership grades, affiliations, manuscript dates, funding, manuscript ID in the page header, and author biographies (omitted).
9. Equations are standard Word equations; the preview I used cannot draw them, so open the file in Word and check Eqs. (1)–(17).

## Update: references and Table XIV
- References increased from 25 to 40. New entries: U-Net, AdamW, SGDR, layer normalization, GELU, mixed-precision training, PyTorch, haze-lines/SQUID (Berman et al.), ResNet, capsule routing, VGG, batch normalization, group normalization, CycleGAN, checkerboard artifacts. Each is cited where the text or a table uses that method.
- All references are renumbered in order of first citation (IEEE style); ranges such as [5]–[7] are kept. Table I now cites SQUID and SUIM-E.
- Table XIV: † marks values where Deeper X v2 is better than that published model (fewer parameters or GFLOPs; higher PSNR or SSIM). PSNR/SSIM marks are given only against our result on the same dataset family (EUVP, UIEB, LSUI-400), and the footnote says the test subsets differ. Rows stay ordered by parameter count. Where a model beats us (e.g. PSNR of Shallow-UWnet, FUSION, DGC-UWnet) there is no mark.
- Please verify the 15 new reference entries (volume, pages, year) against the original sources before submission. I could not look them up online here. The Table XIV models (DGC-UWnet, LSNet, USUIR, FUSION, snnTrans-DHZ) and the SUIM-E, U45 and RUIE sources still need your own citations.
