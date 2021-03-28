# General

Max pts: 100 (Complete route without intervention)

Max penalty pts: -100 (Collision)

# Penalty

## Safety

- -100 Collision (Debounced)
  
## Traffic Rules

- -20  Lane invasion (DoubleSolid)
- -15  Running red light (WIP)
- -10  Lane invasion (Solid)

## Route

- -5 Reroute

## Speed

- -10 Higher than speed limit (Debounced)

# Reward

## Route

- +100 Complete Route

## Forward

- +0.1 Forward 1 meter
