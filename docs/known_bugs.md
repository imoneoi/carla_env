# Known Bugs

## 1. Some collision sensor not working
https://github.com/carla-simulator/carla/issues/1553

https://github.com/carla-simulator/carla/issues/4038

Reason: Car models defects, only following cars have this issue:
```
vehicle.chargercop2020.chargercop2020
vehicle.charger2020.charger2020
vehicle.mercedesccc.mercedesccc
```

Solution: Blacklist these cars


## 2. dt gap? simulator dt != real dt

## 3. Memory Leakage

Solution: limit the server lifetime < 10