"""
Extension classes enhance TouchDesigner components with python. An
extension is accessed via ext.ExtensionClassName from any operator
within the extended component. If the extension is promoted via its
Promote Extension parameter, all its attributes with capitalized names
can be accessed externally, e.g. op('yourComp').PromotedFunction().

Help: search "Extensions" in wiki
"""

from TDStoreTools import StorageManager
import TDFunctions as TDF

class CameraFX:
	"""
	CameraFX description
	"""
	def __init__(self, ownerComp: baseCOMP):
		self.ownerComp = ownerComp

		# properties
		TDF.createProperty(self, 'MyProperty', value=0, dependable=True,
						   readOnly=False)

		# attributes:
		self.a = 0 # attribute
		self.B = 1 # promoted attribute

		# stored items (persistent across saves and re-initialization):
		storedItems = [
			# Only 'name' is required...
			{'name': 'StoredProperty', 'default': None, 'readOnly': False,
			 						'property': True, 'dependable': True},
		]
		# Uncomment the line below to store StoredProperty. To clear stored
		# 	items, use the Storage section of the Component Editor
		
		# self.stored = StorageManager(self, ownerComp, storedItems)

	def myFunction(self, v):
		debug(v)

	def PromotedFunction(self, v):
		debug(v)

	# def onDestroyTD(self):
	# 	"""
	# 	Called when the extension or component is being deleted. Use this
	# 	instead of __del__ for cleanup tasks.
	# 	"""
	# 	debug("onDestroyTD called")

	# def onInitTD(self):
	# 	"""
	# 	Called after the extension is fully initialized and attached to the 
	# 	component. Use this instead of __init__ for tasks that require other
	# 	components' extensions to be available, or that use promoted members.
	# 	"""
	# 	debug("onInitTD called")