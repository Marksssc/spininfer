from models.ising import IsingModel
from models.potts import PottsModel

Model = IsingModel | PottsModel
