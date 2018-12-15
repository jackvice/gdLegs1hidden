using Knet
using Dates
import Gym
import Random
using DelimitedFiles


function main(;
              mLoop = 100,
              renderPeriod = 2,
              render = true,
              randSeed = 17,
	      btype = Array{Float32}, #no gpu for now
	      #atype = Array{Float32} #no gpu for now
              atype = gpu() >= 0 ? KnetArray{Float32} : Array{Float32}, #(C)
              )
    decayRate = 0.99# 0.99 # decay Rate for RMSProp leaky sum of grad^2
    batchSize = 220 #2 #200  # every how many episodes to do a param update?
    weightUpdate = 20 #1 #20
    learningRate = 1e-3 #1e-3 #
    gamma = 0.9 # 0.99  #discount factor for reward
    runningReward = false
    inputDim = 14
    hiddenSize = 28 #https://www.heatonresearch.com/2017/06/01/hidden-layers.html
    #hiddenSize2 = 12 #https://www.heatonresearch.com/2017/06/01/hidden-layers.html
    outputDim = 24
    rewardSum = 0
    maxReward = 0
    startUnixTime = time()
    cycleNum = 0
    numFalls = 0
    batchNum = 0
    minFallRatio = 99999
    renderUpdate = 0
    printStuff = 0
    reward = 0
    legUpdateRate = 5
    onPolicy = 0 # don't start on policy
    
    ###           initialize arrays
    historyObservations = atype(undef, inputDim, 0) #
    historyFalls = btype(undef, 1, 0) #
    historyRewards = btype(undef, 1, 0) #
    historyPredict = btype(undef, 2, 0) #
    historyWeights = atype(undef, 2, 0) #
    tempGradientSum = btype[ zeros(Float32, inputDim ,hiddenSize),
                             zeros(Float32, hiddenSize, outputDim)]
    historyLossGradient = btype(undef, outputDim, 0) #
    historyHidden = atype(undef, hiddenSize, 0)
    
    weights = atype[ randn(Float32,inputDim,hiddenSize)/sqrt(inputDim),
                     randn(Float32,hiddenSize,outputDim)/sqrt(hiddenSize)]
    expectationGsquared = atype[zeros(inputDim,hiddenSize),
                                zeros(hiddenSize,outputDim)]

    #predictedAction = zeros(outputDim)
    fakeLabels = zeros(outputDim)
    
    env = Gym.GymEnv("BipedalWalker-v2")# (C)
    if randSeed > 0 # This if block is same as previous line but clearer
        Random.seed!(randSeed)
        Gym.seed!(env, randSeed)
    end
    observation = Gym.reset!(env)
    render && Gym.render(env)
    initVelocity = 1.0
    motorAction = [initVelocity initVelocity (-1*initVelocity) (-1*initVelocity)]
    fall = false
    while true
        cycleNum +=1
        
        observation = convert(atype,observation[1:14])
        
        (hiddenValues, predictedAction) = predict(weights, observation, inputDim,
                                                  hiddenSize, outputDim, printStuff)
        historyObservations = [ historyObservations observation ] # append
	historyHidden = [ historyHidden hiddenValues ] # append
        
        # Now we get our motor commands per joint and fake labels
        fakeLabels = motorActionValsLabelsFromProbs(batchNum, motorAction, fakeLabels,
                                                    predictedAction, maxReward,onPolicy)
        reward = 0.0
        tempReward = 0.0 
        for i = 1:legUpdateRate
            observation, tempReward, done, info = Gym.step!(env, motorAction)
            render && Gym.render(env)
            if reward == -100
                reward = (3*maxReward) * -1
                numFalls += 1
                fall = true
                observation = Gym.reset!(env)
                break
            else
                reward += tempReward
            end
        end

        #test of pushing weights
        if false #observation[2] > .01
            fakeLabels = zeros(24)
            fakeLabels[1] = 1
            #fakeLabels[9] = 1
            #fakeLabels[15] = 1
            fakeLabels[24] = 1
            reward = 50
        end
        #end test of pushing 

        rewardSum += reward
        historyRewards = [ historyRewards reward ]
        println("predictedAction: ",predictedAction)
        println("               fakeLabels: ",fakeLabels)
        ##lossGradient = predictedAction - fakeLabels # just to try if it works?
        lossGradient = fakeLabels - predictedAction # 24 of each
        historyLossGradient = [ historyLossGradient  lossGradient ] # append

        if cycleNum % batchSize == 0 #||  fall
            batchNum += 1
            if batchNum % 20 == 0
                #println("observation[2]: ",observation[2])
                (render, printStuff, onPolicy) = readRenderFile("renderPrint.txt",env) 
            end
            
            gradientLogDiscounted = DiscountWithRewards(historyLossGradient',
                                                        historyRewards', gamma)
            #println("historyRewards ", historyRewards[1:20])
            #println("historyLossGradient[1:20] ",historyLossGradient[1:20])
            #println("gradientLogDiscounted: ", gradientLogDiscounted[1:20])
            #Gym.close!(env)
            
            gradientLogDiscounted = convert(atype, gradientLogDiscounted)
            gradient = gradientCalc(gradientLogDiscounted, historyHidden,
                      		    historyObservations, weights)
            
            for i = 1:3
                #println("i:",i," size gradient ", size(convert(btype,gradient[i])))
		tempGradientSum[i] += convert(btype,gradient[i])
	    end
            
	    return  ################################# RETURN

	    if runningReward == false  # first time
		runningReward = rewardSum
	    else
		runningReward = runningReward * 0.99 + rewardSum * 0.01
	    end
            if reward > maxReward
                maxReward = reward
            end
            fallRatio = numFalls/batchNum 
            if fallRatio < minFallRatio
                minFallRatio = fallRatio
            end
            if batchNum % weightUpdate == 0
                writeDataToFile(batchNum, convert(Array{Float32},weights[1]),
                                convert(Array{Float32},weights[2]),
                                rewardSum, runningReward, startUnixTime,
                                fallRatio, predictedAction)
                if printStuff == 1 || printStuff == 2
                    printSomeStuff(predictedAction, fakeLabels)
                    observation = Gym.reset!(env)
                end
                
                weightsUpdate(weights,learningRate, decayRate, expectationGsquared,
                              tempGradientSum)            
                println("Batch: ", batchNum, ",  Cycle: ", cycleNum , ", Reward: ",
                        reward,
                        ", Running fall Ratio: ",  fallRatio)
                println("Running reward: ", runningReward,", Max Reward: ",
                        maxReward, ", Min Fall Ratio: ", minFallRatio)

                println(" ")
            end            
            historyObservations = atype(undef, inputDim, 0) #
	    historyHidden = atype(undef, hiddenSize, 0) #
	    historyLossGradient = btype(undef, outputDim, 0) #
	    historyRewards = btype(undef, 1, 0) #


        end
    end
    render && Gym.close!(env)
    println("history fall episode numbers ", historyFalls )
    return
end

function motorActionValsLabelsFromProbs(batch, action, labels, predicted, maxReward,onPolicy)
    actionVals = [-1.0 -0.666 -0.333 0.333 0.666 1.0]
    maxIndex = [0 0 0 0] #indexs for the max of each set of 
    labels = zeros(24)
    j=1
    #println("              ")
    #println(predicted)
    for i = 1:4
        #find the max value of the each set of 6 values
        maxIndex[i] = findfirst(isequal(maximum(predicted[ j : ( i * 6 ) ] ) ),
                                 predicted[ j : ( i * 6 ) ]) + ((i-1) *6)
        #println("i: ", i,",  maxIndex[i]: ",maxIndex[i],",  j: ",
        #        j,",  max: ", maximum(predicted[ j : ( i * 6 ) ] ) )
        #println("predicted[ j : ( i * 6 ) ] ) ): ",predicted[ j : ( i * 6 ) ] )
        
        j +=6
    end
    #println("tempIndex: ", tempIndex )
    for k = 1:4
        exploreRnd = rand()
        exploreThresh = batch / 100000.0
        if exploreThresh > 0.95 # would be only on policy
            exploreThresh = 0.95 # always explore by at least 5%
        end
        
        #println("x and predicted[ tempIndex[ i ] ]", x , predicted[ tempIndex[ i ] ] )
        if exploreRnd <  exploreThresh || onPolicy == 1
            
            #println("On Policy randAction: ",randAction,",  k: ",
            #        k, ",  tempIndex[k]: ", tempIndex[k],",   (k-1)*6: ", (k-1)*6 ) 
            ##ERROR: BoundsError: attempt to access 1×6 Array{Float64,2} at index [-9]
            ##x: 0.5732682297117717,  i: 4,  tempIndex: ,  tempIndex[i]: 7,   (i-1)*6: 18
            ##ERROR: BoundsError: attempt to access 1×6 Array{Float64,2} at index [-11]
            action[k] = actionVals[maxIndex[k]-((k-1)*6)]
            labels[maxIndex[k]] = 1.0
        else #take random action
            #action[k] = actionVals[tempIndex[ k ]-(( k - 1 ) * 6 )] * -1
            #tempFound = findfirst(isequal(action[ k ]), actionVals )
            #labels[tempFound[2] + (( k - 1 ) * 6 )] = 1.0
            randAction = rand(1:6) 
            action[k] = actionVals[randAction]
            labels[((k-1) * 6) + randAction] = 1.0
            #println("IsRand, explore: ", explore,",  k: ",
            #        k, ",  maxIndex[k]: ", maxIndex[k],", action[k]: ",action[k]  ) 
        end
    end
    #println("labels: ",labels)
    return labels
end




#take a probability and return a int action using random
function getActionFromProb(probAction,maxTorque)
    x = rand()
    if x < probAction
        return maxTorque #forward
    else
    	return -1*maxTorque #back
    end
end


rnd() = 2 * rand() - 1


#fix to knet in here.
function weightsUpdate(weights,learningRate, decayRate, expectationGsquared, gBatchSum)
    epsilon = 1e-5
    for i = 1:3 #based on the number of layers
	tempGradient = gBatchSum[i]
        
        if false #true# i == 2
            println("i is ",i) #," tempGradient: ",tempGradient)
            testA1 = decayRate * convert(Array{Float32},expectationGsquared[i])
            testA2 = tempGradient.^2
            testA3 = (1-decayRate) * tempGradient.^2
            println("testA1 size", size(testA1))
            println("testA3 size", size(testA3))
            testA4 = decayRate * convert(Array{Float32},expectationGsquared[i]) +
                ((1-decayRate) * tempGradient.^2)
            println("testA4 size", size(testA4))
        end
	expectationGsquared[i] = decayRate * convert(Array{Float32},expectationGsquared[i]) +
            ((1-decayRate) * tempGradient.^2)
	#println("size tempGradient[i]",size(tempGradient))
	#println("size tempGradient",size(tempGradient))
	#println("e ",e)
	z1 = convert(Array{Float32},(learningRate * tempGradient)) # make knet later
	z2 = convert(Array{Float32},(sqrt.(expectationGsquared[i] .+ epsilon)))
        #z1 = convert(KnetArray{Float32},learningRate * tempGradient) # make knet later
	#z2 = sqrt.(expectationGsquared[i] .+ e)
	#println("size z1",size(z1))
	#println("size z2",size(z2))
	z3 = z1 ./ z2
        z3 = convert(KnetArray{Float32},z3)
	#println("size z3",size(z3))
        
        weights[i] += z3

	#weights[i] += (learningRate * tempGradient)./(sqrt.(expectationGsquared[i] .+ e))
	gBatchSum[i] = zeros(Float32, size(weights[i])) #reset the batch gradient buffer (C)
    end
end


function gradientCalc(gradientLogDiscounted, historyHidden,
                      historyObservations, weights)
    deltaLog = gradientLogDiscounted
    #println("\n")

    #println("historyHidden2 size ", size(historyHidden2))
    #println("gradientLogDiscounted and deltaLog size ", size(deltaLog))
    #println("DCost_DWeight2 = (deltaLog * historyHidden')")
    #println("\n")
    
    DCost_DWeight3 = historyHidden2 * deltaLog

    deltaLog3 = deltaLog * weights[3]'
    deltaLog3 = relu.(deltaLog3)
    #println("size(historyHidden1): ",size(historyHidden1))
    #println("size(deltaLog3): ",size(deltaLog3))
    DCost_DWeight2 = (historyHidden1 * deltaLog3)
    ##DCost_DWeight2 = dot(historyHidden', deltaLog)

    #println("size(weights[2]): ",size(weights[2]))
    #println("deltaLog2 = deltaLog' * weights[2]")
    
    
    deltaLog2 = deltaLog3 * weights[2]'
    #println("                                     size(deltaLog2) ",size(deltaLog2))
    #println("                                     size(historyObservations ",size(historyObservations))
    deltaLog2 = relu.(deltaLog2)
    DCost_DWeight1 = (historyObservations * deltaLog2)
    ##DCost_DWeight1 = deltaLog2' * historyObservations 
    #println(" size(DCost_DWeight1) ",size(DCost_DWeight1))
    #println(" size(DCost_DWeight2) ",size(DCost_DWeight2))
    #println("\n")
    return (DCost_DWeight1, DCost_DWeight2)
end

function gradientCalcOld(gradientLogDiscounted, historyHidden, historyObservations, weights)
    deltaLog = gradientLogDiscounted

    DCost_DWeight2 = historyHidden * deltaLog
    
    deltaLog2 = deltaLog * weights[2]'

    deltaLog2 = relu.(deltaLog2)

    DCost_DWeight1 = (historyObservations * deltaLog2)

    return (DCost_DWeight1', DCost_DWeight2)
end

mean(x) = sum(x) / length(x)
std(z) = sqrt(mean(map(x -> (x - mean(z))^2, z)))

function DiscountWithRewards(historyLossGradient, historyRewards, gamma)
    returnVal = zeros(220,24)
    discountEpisodeRewards = discountRewardsFall(historyRewards, gamma)
    #println("DiscountedEpisodRewards: ", discountEpisodeRewards)
    #rewardsDiscounted = map( x-> x - mean(rewardsDiscounted), rewardsDiscounted)
    discountEpisodeRewards = discountEpisodeRewards .- mean(discountEpisodeRewards)
    discountEpisodeRewards = discountEpisodeRewards ./ std(discountEpisodeRewards)
    #rewardsDiscounted = map( x-> x / std(rewardsDiscounted), rewardsDiscounted)
    #println(historyLossGradient[1,:])
    #println("                  discountwithrewards          my two shapes",
    #        size(historyLossGradient), size(discountEpisodeRewards))
    #println("shape from discound with rewards", size(historyLossGradient .* discountEpisodeRewards))
    #println("length: ", length(discountEpisodeRewards))
    #return (historyLossGradient .* discountEpisodeRewards) * -1 # I don't know why wrong sign.
    test1 = (historyLossGradient .* discountEpisodeRewards)# I don't know why wrong sign.
    for i = 1:220
        returnVal[i,:] =  historyLossGradient[i,:] * discountEpisodeRewards[i]
    end
    #println(returnVal[1,:], size(returnVal))
    #println(test1[1,:], size(test1))
    return returnVal
end

function discountRewardsFall(rewards, gamma)
    #posThresh = 0.2
    #fall = -10.0
    rewardsDiscounted = zeros(size(rewards))
    tempAdd = 0.0
    for i = length(rewards):-1:1
        #if rewards[i] > posThresh || rewards[i] < fall  #if we have some reward
        #    tempAdd = 0.0
        #end
        tempAdd = tempAdd * gamma + rewards[i]
        rewardsDiscounted[i] = tempAdd
    end
    #println("rewards: ", rewards)
    #println("rewardsDiscounted: ", rewardsDiscounted)
    return rewardsDiscounted
end

function discountRewards(rewards, gamma)
    rewardsDiscounted = zeros(size(rewards))
    tempAdd = 0.0
    for i = length(rewards):-1:1
        if rewards[i] != 0.0  #if we have some reward
            tempAdd = 0.0
        end
        tempAdd = tempAdd * gamma + rewards[i]
        rewardsDiscounted[i] = tempAdd
    end
    #println("rewards: ", rewards)
    #println("rewardsDiscounted: ", rewardsDiscounted)
    return rewardsDiscounted
end


sigmoid(z) = 1.0 ./ (1.0 .+ exp(-z))


function predict(weights, observation,inputDim, hidden1Size, hidden2Size, outputDim, printStuff)
    outputs = zeros(outputDim)

    #println("size weights[1]: ", size(weights[1]), "size observation: ", size(observation))
    hiddenLayerValues = weights[1]' * reshape(observation, inputDim, 1 )

    hiddenLayerValues = relu.(hiddenLayerValues)
    

    #println("size reshape(observation, inputDim, 1 ) :", size(reshape(observation, inputDim, 1 )))
    #println("size weights[2]: ", size(weights[2]), "size : hiddenLayerValues", size(hiddenLayerValues))

    hiddenLayer2Values = weights[2]' * hiddenLayerValues

    hiddenLayer2Values = relu.(hiddenLayer2Values)
    #println("size weights[3]: ", size(weights[3]), ",  size hidden2: ", size(hiddenLayer2Values))
    #println("size reshape(weights[3], outputDim, hidden2Size)  :", size(reshape(weights[3], outputDim, hidden2Size)))
    #outputLayerValues = reshape(weights[3], outputDim, hidden2Size) * hiddenLayer2Values
    outputLayerValues = weights[3]' * hiddenLayer2Values
    
    for i = 1:outputDim
        outputs[i] = sigmoid(outputLayerValues[i])
    end
    #if printStuff == 2
    #    println(outputs)
    #end
    #outputs = convert(Array{Float32},outputLayerValues )
    #j=1
    #for k = 1:4
        #outputs[ j : ( k * 6 ) ] = softmax( outputs[ j : ( k * 6 ) ] )
        #println("i: ", i,",  tempIndex[i]: ",tempIndex[i],",  j: ",j,",
        #   max: ", maximum(predicted[ j : ( i * 6 ) ] ) )
        #println("predicted[ j : ( i * 6 ) ] ) ): ",predicted[ j : ( i * 6 ) ] )
        #j +=6
    #end
    return (hiddenLayerValues, hiddenLayer2Values, outputs)
end



function printObservations(observation)
    observationTypes = ["Hull Angle: " "Hull Angular Velocity: " "Velocity x: " "Velocity y: " "Hip Joint 1 Angle: " "Hip Joint 1 Angle:" "Knee Joint 1 Angle: " "Knee Joint 1 Angle: " "Leg 1 ground Contact: " "Hip Joint 2 Angle: " "Hip Joint 2 Angle:" "Knee Joint 2 Angle: " "Knee Joint 2 Angle: " "Leg 2 ground Contact: " "Lazer 1: " "Lazer 2: " "Lazer 3: " "Lazer 4: " "Lazer 5: " "Lazer 6: " "Lazer 7: " "Lazer 8: " "Lazer 9: " "Lazer 10: "]   
    for i = 1:length(observation)
        println(observationTypes[i], observation[i])
    end
end

function printTime(str,oldTime, maxTime)
    diffTime = time() - oldTime
    if diffTime > 1
        println("time of ",str, diffTime)
    end 
    return (time(), maxTime)
end

function printTimeold(str,oldTime, maxTime)
    diffTime = time() - oldTime
    if diffTime > maxTime
        println("time of ",str,diffTime)
        return (time(), diffTime)
    else
        return (time(), maxTime)
    end
end




function writeDataToFile(batchNum, weights1,weights2, rewardSum, runningReward, startTime, fallRatio, outputs)
    writeWeightsPeriod = 5000
    dataFileName = "dataLegs/dataLegs.csv"
    dateStr = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    weightsFileName = string("dataLegs/legs-",string(batchNum), "weights", dateStr, ".txt" ) 

    if batchNum % writeWeightsPeriod == 0
        open(weightsFileName, "w") do io
            writedlm(io, [weights1,weights2], ',')
        end
    end
    open(dataFileName, "a") do io
        writedlm(io, [dateStr (time()-startTime) batchNum rewardSum runningReward fallRatio], ',')
        #writedlm(io, outputs,',')
      
    end
end

 
function printSomeStuff(predictedAction, fakeLabels)
    #println("fakeLabels: ", fakeLabels)
    println("predictedAction 1-6: ", predictedAction[1:6])
    #println("        fakeLabels 1-6: ", fakeLabels[1:6])
    println("predictedAction 7-12: ", predictedAction[7:12])
    #println("        fakeLabels 7-12: ", fakeLabels[7:12])
    println("predictedAction 13-18: ", predictedAction[13:18])
    #println("        fakeLabels 13-18: ", fakeLabels[13:18])
    println("predictedAction 19-24: ", predictedAction[19:24])
    #println("        fakeLabels 19-24: ", fakeLabels[19:24])
end


function readRenderFile(fileName, env)
        dataUpdate = readdlm(fileName)
        renderUpdate = convert(Int, dataUpdate[1])
        if renderUpdate == 1
            render = true
        else
            render = false
            Gym.close!(env)
        end
        return(render, convert(Int, dataUpdate[2]), convert(Int, dataUpdate[3]))

end

function input(prompt::String="")::String
    print(prompt)
    return chomp(readline())
end
