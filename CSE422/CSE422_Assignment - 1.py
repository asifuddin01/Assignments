# Part 1




import heapq

def finder(inputfile): # maze size read


  with open (inputfile) as i:

    r,c = map (int, i.readline().split())

    sp = tuple ( map (int, i.readline().split()))
    ep = tuple ( map (int, i.readline().split()))

    maze_build = []
    for p in range(r):
      point = i.readline().strip()
      maze_build.append(list(point))




  def heuristic (position):
    return abs (position[0] - ep[0]) + abs (position[1] - ep[1])



  heap = [ (heuristic(sp), 0, sp, "" )]
  visit = { sp:0}


  while heap:
    j,path_cost, position , path = heapq.heappop(heap)

    if position == ep:
      return path_cost, path


    for dx, dy, move in [(-1,0,"U"),(1,0,"D"),(0,-1,"L"),(0,1,"R")]:

      x = position[0]+dx
      y = position[1]+dy


      if 0<=x<r and 0<=y<c and maze_build [x][y]=="0":
        nCost = path_cost+1


        if (x,y) not in visit or nCost < visit[(x,y)]:
          visit[(x,y)] = nCost

          heapq.heappush(heap,( nCost+heuristic((x,y)), nCost, (x,y), path+move))




  return -1, None


path_cost, path = finder("/content/drive/MyDrive/CSE422_Assignment 1/Assignment01_(A_ Search)_InputFile_Part1.txt")
print(path_cost)

print (path if path else -1)

#

# Part 02



def admisibility_checker(inputfile):

  with open (inputfile, "r") as i:

    r,c = map (int, i.readline().split())
    strt , gl = map (int, i.readline().split())

    h_val= {}



    for o in range (r):

      vertices = i.readline().split()
      n = int(vertices[0])
      heuristic = int(vertices[1])
      h_val[n] = heuristic





    graph = [[] for j in range (r+1)]

    for k in range (c):
      x, y = map (int, i.readline().split())
      graph[x].append(y)
      graph[y].append(x)




    dis = [float('inf')] * (r+1)
    dis[gl]= 0
    que = [gl]



    while que :
      c = que.pop(0)

      for n in graph [c]:
        if dis[n]==float('inf'):
          dis[n]= dis[c] +1
          que.append(n)

    far_n = []

    for l in range(1,r+1):
      if h_val[l] > dis[l]:
        far_n.append(l)

    if not far_n:
      print("1")
      print("The heuristic values are admissible")

    else:
      print("0")
      print("Here Nodes With","".join(map(str,far_n)), "are inadmissible")





admisibility_checker("/content/drive/MyDrive/CSE422_Assignment 1/Assignement01_(A- Search)_InputFile_Part2 - Copy.txt")

