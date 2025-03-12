## Add gaussian noise to tensor

- Go to `train` section under `transforms` section in the yaml file, add:
```
  - ${gaussian_noise}
```
in the appropriate order with the other transforms. Hydra should take care of the rest.
