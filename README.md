# Drawing-Aware PD detection

### Introduction

This repository contains an open-source toolbox for Parkinson's Disease (PD) detection from handwriting records based on PyTorch.

The main branch works with **PyTorch 1.8+**.

### What's New

* Release **Drawing-type classifier** ([code](https://github.com/IoannisKansizoglou/Drawing-Aware_PD_detection_through_hierarchical_DL_models/blob/main/codes/drawing-type%20classifier/train_model.py) and [weights](https://duth-my.sharepoint.com/:f:/g/personal/ikansizo_duth_gr/EohrhPpey5ZPhvJEhRi4wggBQRegu0VzsakZZ-SRll6MFA?e=uKOR5N)): a ResNet-18 model for classifying different drawing types (see drawing types below from the [NewHandPD](https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/) database).

<table align="center">
  <tr>
    <td align="center">
      <img src="images/Circle1.png" width="100" alt="Image 1"/>
      <img src="images/Circle2.png" width="100" alt="Image 2"/><br/>
      <sub> <em>(a) Circle</em> </sub>
    </td>
    <td align="center">
      <img src="images/Meander1.png" width="100" alt="Image 3"/>
      <img src="images/Meander2.png" width="100" alt="Image 4"/><br/>
      <sub> <em>(b) Meander</em> </sub>
    </td>
    <td align="center">
      <img src="images/Spiral1.png" width="100" alt="Image 5"/>
      <img src="images/Spiral2.png" width="100" alt="Image 6"/><br/>
      <sub> <em>(c) Spiral</em> </sub>
    </td>
  </tr>
</table>
<br>

*  Release **Drawing-aware PD detectors** (code and [weights](https://duth-my.sharepoint.com/:f:/g/personal/ikansizo_duth_gr/Eh3UtjNP4iFLmj4XJRO98pwBgdp7XM96wyUJv9woAX4uXw?e=HV5hs9)): individual ResNet-18 models for PD detection from each drawing-type.

### Proposed Method



<p align="center">
  <img align="middle" src="images/Hierarchical Architecture.png" width="75%"/>
</p>
