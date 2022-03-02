from data import *
from slam import *
from pr2_utils import *
from lidar import *
from helpers import *
# from Map2D import *
# data = data()
pkl_file = open('data.pkl', 'rb')
data = pickle.load(pkl_file)
slam = slam(50, (-300, 1400), (-1100, 800), 1)
theta, x_pred, y_pred = 0, 0, 0
noiseSigma = np.diag([5e-2, 5e-2, 5e-2])
lidar = lidar(slam.mapLogOdds)
# map = map
lidarInWorldParticles = np.empty((slam.numParticles,286,2))
pred=[]
trust = math.log(9)
logOddsLim = (-100, 100)
x_im = np.arange(slam.mapLogOdds.MAP['xmin'],slam.mapLogOdds.MAP['xmax']+slam.mapLogOdds.MAP['res'],slam.mapLogOdds.MAP['res']) #x-positions of each pixel of the map
y_im = np.arange(slam.mapLogOdds.MAP['ymin'],slam.mapLogOdds.MAP['ymax']+slam.mapLogOdds.MAP['res'],slam.mapLogOdds.MAP['res']) #x-positions of each pixel of the map
for i in range(1, len(data.lidar_data[:,0])):

    # get lidar poiints
    lidarInV,  lidarInS= lidar.getLidarInV(data.lidar_data[i]) # 286 by 2
    lidarInW = np.array([x_pred,y_pred]) + lidarInV

    # build binary map
    lidarInM = slam.mapLogOdds.coordToMap(lidarInW)
    if i==1:
        slam.mapLogOdds.MAP['map'][lidarInM[0], lidarInM[1]]=1

    # prediction
    time_diff1 = (data.encoder_timestamp[i] - data.encoder_timestamp[i-1])*(10**(-9))
    time_diff2 = (data.fog_timestamp[i] - data.fog_timestamp[i-1])*(10**(-9))
    left_velocity, right_velocity = (data.left_meters[i]-data.left_meters[i-1])/(time_diff1) , (data.right_meters[i]-data.right_meters[i-1])/(time_diff1)
    lin_vel =  (((left_velocity+right_velocity) / 2.0))
    tau = time_diff1
    theta += tau* (data.fog_data[i-1][-1]  / tau)
    x_pred += (tau*lin_vel* (math.cos(theta)) )
    y_pred += (tau*lin_vel* (math.sin(theta)) )
    # pred.append((x_pred, y_pred))

    # add noise, get particles
    u = np.array([x_pred, y_pred, theta])
    u = np.random.multivariate_normal(np.array(u), noiseSigma, slam.numParticles) # 5 by 3
    uxy = u[:,:-1]

    # skippage
    if i%100!=0: continue
    print(i)

    # get lidar points for each particle
    for j in range(slam.numParticles):
        lidarInWorldParticles[j] = uxy[j]+lidarInV #np.sum(uxy[i], lidarInWorld) # 5 by 286 by 2
        # print("shape1",lidarInWorldParticles[j].shape)
        lidarInWorldParticles[j] = lidarInWorldParticles[j][slam.mapLogOdds.in_map(lidarInWorldParticles[j]),:]
        # print("shape2",lidarInWorldParticles[j].shape)

    # run map correlation for each particle
    mapCorrelOut=[]
    Cnorm = 0
    for p in range(slam.numParticles):
        runMapCorrelVectorized(u[p], lidarInV, slam.mapLogOdds.MAP['map'], slam)
        # maxc, idx, particle = runMapCorrel(p, uxy[p,0], uxy[p,1], x_im, y_im, lidarInS, slam.mapLogOdds.MAP['map'], slam.bias, False)
        # Cnorm += (slam.weights[p]*maxc)
        break
    break

        # mapCorrelOut.append((maxc,idx, particle))
    if Cnorm==0:
        print("Cnorm is 0")
        # continue
    # get max from map correlation output NOT NEEDED
    # Cstar, istar, maxparticle = max(mapCorrelOut, key= lambda x: x[0])
    
    # update particle pose NOT NEEDED
    # x_pred, y_pred = uxy[particle,0] + slam.bias[istar][0], uxy[particle,1] + slam.bias[istar][1]
    # x_pred, y_pred = uxy[maxparticle,0], uxy[maxparticle,1]

    # update particle weights
    for  k in range(len(slam.weights)):
        slam.weights[k]*= (mapCorrelOut[k][0] / Cnorm)
    # print(np.sum(slam.weights))
    # print(slam.weights[0])


    # update map based on max weight particle
    largestWeightIdx = np.argmax(slam.weights)
    particlepose = uxy[largestWeightIdx]
    lidar_tmp_inW = np.array([particlepose[0],particlepose[1]]) + lidarInV
    for k in range(len(lidar_tmp_inW[:,0])):
        bres= slam.mapLogOdds.coordToMap(bresenham2D(particlepose[0],particlepose[1],lidar_tmp_inW[k,0],lidar_tmp_inW[k,1]).T).T
        bresend = bres[:,-1].astype(int)
        brespoints = bres[:,:-1].astype(int)
        np.add.at(slam.mapLogOdds.MAP['mapLogOdds'],[brespoints[0], brespoints[1]],-trust)
        slam.mapLogOdds.MAP['mapLogOdds'][bresend[0], bresend[1]]+=trust
        # if slam.mapLogOdds.MAP['mapLogOdds'][bresend[0], bresend[1]]>0:
        #     slam.mapLogOdds.MAP['map'][bresend[0], bresend[1]]=1
        # else:
        #     slam.mapLogOdds.MAP['map'][bresend[0], bresend[1]]=0
    slam.mapLogOdds.MAP['mapLogOdds'] = np.clip(slam.mapLogOdds.MAP['mapLogOdds'], *logOddsLim) # clip logodds prevent overconfidence
    
    # update pose based on max weight particle
    x_pred, y_pred = particlepose[0], particlepose[1]
    pred.append((x_pred, y_pred))



    # theta = u[particle,2]


        
   
slam.mapLogOdds.MAP['map'] = (slam.mapLogOdds.MAP['mapLogOdds'] > 0).astype(int)  
 
fig1 = plt.figure()
plt.imshow(slam.mapLogOdds.MAP['map'],cmap="gray")
plt.title("Occupancy grid map")
plt.show(block=True)    

trust = math.log(9)
fig3 = plt.figure()
plt.scatter([x[0] for x in pred],[y[1] for y in pred] )
plt.show(block=True)
    
