
from TDStoreTools import StorageManager
import TDFunctions as TDF

from AppStore import AppStore
from App import App

class CameraFX:
	"""
	CameraFX description
	"""
	def __init__(self, ownerComp: baseCOMP):
		self.ownerComp = ownerComp

		# grab important nodes
		self.YOLO26_SEG: baseCOMP = self.ownerComp.op('YOLO26_SEG')

		# set default state
		self.YOLO26_SEG.allowCooking = True
 