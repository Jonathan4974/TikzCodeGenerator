# UltraSketch Evaluation Summary

> Note: pixel-level CC only. Used as a relative comparison signal across prompt variants, not as an absolute score comparable to TikZero's reported (SigLIP-based) CC.

### Pixel-level CC (UltraSketch output)

| Variant                | Mean CC | Std CC |
|:-----------------------|--------:|-------:|
| baseline               |   0.932 |  0.044 |
| pencil                 |   0.929 |  0.050 |
| scientific             |   0.931 |  0.046 |
| minimal                |   0.933 |  0.042 |
| structure_preserve     |   0.923 |  0.058 |
| scientific_labels      |   0.931 |  0.048 |
| light_pencil           |   0.926 |  0.042 |
| anti_artifact          |   0.921 |  0.061 |
| student_notes          |   0.927 |  0.056 |


### Pixel-level CC (displacement-only baseline)

| Variant                | Mean CC | Std CC |
|:-----------------------|--------:|-------:|
| displacement           |   0.928 |  0.063 |


### Pixel-level CC (saved UltraSketch + displacement visual blend)

| Variant                | Mean CC | Std CC |
|:-----------------------|--------:|-------:|
| baseline               |   0.934 |  0.054 |
| pencil                 |   0.933 |  0.057 |
| scientific             |   0.933 |  0.053 |
| minimal                |   0.935 |  0.051 |
| structure_preserve     |   0.932 |  0.058 |
| scientific_labels      |   0.933 |  0.055 |
| light_pencil           |   0.933 |  0.054 |
| anti_artifact          |   0.930 |  0.061 |
| student_notes          |   0.932 |  0.058 |

### Conclusion

No real difference in doing some prompt engineering here..
