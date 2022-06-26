# Deep Tweezer

This repository keeps the code regarding a research project developed by GOL (Guerreiro's Optomechanics Laboratory). 
Deep Tweezer is (INSERT FULL DESCRIPTION HERE)

* Feature n
* Feature n
* Feature n
* Feature n

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install pymentor.

```bash
make install
```

## Usage

```bash
conda activate deep-tweezer
```

```bash
make help
```


```python
import numpy as np
from pymentor import Mentor
# direct kinematics example
angles = [np.pi/6]*5
robot = Mentor()
pos, rot = robot.get_position(angles)
# pos is 4x1 vector 
# rot is 3x3 rotation matrix
       
pos = np.array([24.21323027, 13.97951501, -17.07885504, 1.0])
rot = np.array([[0.59049287, 0.23642905, -0.77163428],
    [-0.23642905, -0.86349762, -0.44550326],
    [-0.77163428, 0.44550326, -0.4539905 ]])
# pos is 4x1 vector 
# rot is 3x3 rotation matrix
angles = robot.get_angles(pos,rot)


# creating rotational matrix from alpha-beta-gamma angles
rot = robot.get_orientation(np.pi/6, np.pi/6, np.pi/6)
```

## Contributing

Any contributions you make are **greatly appreciated**. To contribute please follow this steps:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/new_feature`)
3. Commit your Changes (`git commit -m 'commit-tag: commit-description'`)
4. Push to the Branch (`git push origin feature/new_feature`)
5. Open a Pull Request

## License
General Public License version 3.0 [GPL-3.0](https://choosealicense.com/licenses/gpl-3.0/)

## Contact

Oscar Schmitt Kremer - [Linkedin](https://www.linkedin.com/in/oscar-kremer/) [Email](oscarkremer97@gmail.com)

Project Link: [Deep Tweezer Repository](https://github.com/oscarkremer/deep-tweezer)

## References



Author 1, Author 2, Author 3. *Article title* **Journal**. Year. [DOI](DOI)