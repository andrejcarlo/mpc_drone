from dataclasses import dataclass


@dataclass
class Drone:
    I_x: float
    I_y: float
    I_z: float
    l: float
    I_r: float
    k_f: float
    k_m: float
    m: float
    g: float
    k_tx: float
    k_ty: float
    k_tz: float
    k_rx: float
    k_ry: float
    k_rz: float
    w_r: float


# Drone from pybullet gym assets
drone_cf2x = Drone(
    I_x=1.4e-5,
    I_y=1.4e-5,
    I_z=2.17e-5,
    l=0.0397,
    I_r=6e-05,
    k_f=3.16e-10,
    k_m=7.94e-12,
    m=0.027,
    g=9.80665,
    k_tx=0,
    k_ty=0,
    k_tz=0,
    k_rx=0,
    k_ry=0,
    k_rz=0,
    w_r=0,
)

### Drone from  M Islam et al 2017 IOP Conf. Ser.: Mater. Sci. Eng. 270 012007
drone_m_islam = Drone(
    I_x=7.5e-3,
    I_y=7.5e-3,
    I_z=1.3e-2,
    l=0.23,
    I_r=6e-05,
    k_f=3.13e-5,
    k_m=7.5e-7,
    m=0.65,
    g=9.80665,
    k_tx=0.,
    k_ty=0.,
    k_tz=0.,
    k_rx=0.,
    k_ry=0.,
    k_rz=0.,
    w_r=0.,
)
