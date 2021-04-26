import shutil, os
class ArtifactManager():
    def __init__(self):
        self.setArtifactContainer("default")
    
    def getDir(self):
        return "artifacts/" + self.getArtifactContainer() + "/"

    def getArtifactContainer(self):
        return self.currentArtifactContainer

    def setArtifactContainer(self, containerName):
        if (os.path.exists("artifacts/" + containerName) == False):
            self.newArtifactContainer(containerName)

        self.currentArtifactContainer = containerName

    def newArtifactContainer(self, containerName):
        if (os.path.exists("artifacts/" + containerName)):
            shutil.rmtree("artifacts/" + containerName)
        os.makedirs("artifacts/" + containerName)

artifact_manager = ArtifactManager()