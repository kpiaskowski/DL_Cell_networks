# convolutional
img_w_conv = 484
img_h_conv = 484
label_w_conv = 14
label_h_conv = 14
batch_size_conv = 10

C = 80

id_to_class = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck',
               9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
               16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
               24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
               34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
               40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
               46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
               54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
               61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet',
               72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave',
               79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase',
               87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

class_to_id = {'backpack': 27, 'spoon': 50, 'sports ball': 37, 'cat': 17, 'tie': 32, 'sheep': 20, 'elephant': 22,
               'toaster': 80, 'clock': 85, 'motorcycle': 4, 'broccoli': 56, 'skateboard': 41, 'bus': 6, 'laptop': 73,
               'horse': 19, 'person': 1, 'fire hydrant': 11, 'couch': 63, 'sandwich': 54, 'baseball bat': 39,
               'refrigerator': 82, 'carrot': 57, 'boat': 9, 'airplane': 5, 'skis': 35, 'giraffe': 25, 'dog': 18,
               'dining table': 67, 'surfboard': 42, 'frisbee': 34, 'bear': 23, 'hot dog': 58, 'baseball glove': 40,
               'toothbrush': 90, 'train': 7, 'bench': 15, 'snowboard': 36, 'mouse': 74, 'oven': 79, 'vase': 86,
               'bicycle': 2, 'kite': 38, 'cow': 21, 'wine glass': 46, 'cup': 47, 'traffic light': 10,
               'parking meter': 14, 'microwave': 78, 'umbrella': 28, 'zebra': 24, 'teddy bear': 88, 'cell phone': 77,
               'sink': 81, 'bottle': 44, 'apple': 53, 'truck': 8, 'chair': 62, 'scissors': 87, 'fork': 48,
               'handbag': 31, 'donut': 60, 'bird': 16, 'tv': 72, 'cake': 61, 'potted plant': 64, 'knife': 49,
               'hair drier': 89, 'book': 84, 'bowl': 51, 'remote': 75, 'bed': 65, 'suitcase': 33, 'stop sign': 13,
               'pizza': 59, 'car': 3, 'tennis racket': 43, 'toilet': 70, 'orange': 55, 'keyboard': 76, 'banana': 52}

colors = [[0.6946012909016442, 0.22203788921925638, 0.22231489274081917],
          [0.2453272549480272, 0.6298809433531001, 0.8444469061243758],
          [0.5532260212458089, 0.4823342765273143, 0.8371946527276047],
          [0.5391321350830178, 0.8726647981298272, 0.8000178253064414],
          [0.42858758644179173, 0.172806533771792, 0.9541887417812738],
          [0.5196181531566046, 0.6787952567945763, 0.17606675121032467],
          [0.8442683716304472, 0.12395654220883223, 0.8457170336436515],
          [0.9057988364508788, 0.005740754321964081, 0.0001324501393724642],
          [0.5540597072853505, 0.9993136803200169, 0.5174955463884289],
          [0.2556234322262281, 0.23791439431078032, 0.27779794734547525],
          [0.8484301734779764, 0.6073600370922727, 0.5993184885093886],
          [0.1263130575734439, 0.4547012643606756, 0.9528867463709803],
          [0.6101862747687096, 0.09179680100253473, 0.20284304070289427],
          [0.36253657248727866, 0.8232185655728167, 0.8962066595261041],
          [0.5627074234889748, 0.309103113232569, 0.2698709848912809],
          [0.5061086231681807, 0.4691448155122362, 0.8004963757024571],
          [0.1180868772349083, 0.9715033320519528, 0.15668232111701552],
          [0.13666504815833802, 0.6794726084136796, 0.5558222573452442],
          [0.03455291980251307, 0.11878838021909244, 0.0826338112700179],
          [0.09887015972763058, 0.7993385802738223, 0.04652895559567383],
          [0.9676582754421882, 0.3478357249526215, 0.42951270460290814],
          [0.5401632609616566, 0.1835601751181467, 0.009277738108685152],
          [0.06416818056398632, 0.34853636462211834, 0.9017665319651265],
          [0.9760573268240627, 0.6240057963366037, 0.38524242989475943],
          [0.49106608069794766, 0.7479781509803842, 0.5911754595193505],
          [0.15694294470373016, 0.08561335777520818, 0.34534574655762873],
          [0.7891777734105982, 0.7479229407936542, 0.6751906102058055],
          [0.7953415346317443, 0.28986257415901306, 0.35175340848948056],
          [0.47580291402701413, 0.9347045776953089, 0.25094668876472914],
          [0.4802473972836194, 0.7722695547526223, 0.3424667040076026],
          [0.7182260802951722, 0.3524064209904676, 0.01809160421782552],
          [0.6455922487352044, 0.3176597919072476, 0.7122011352597544],
          [0.1480411947872624, 0.9879887816440107, 0.9152134640628499],
          [0.4556918004987811, 0.10793398670425303, 0.47420042732341827],
          [0.512406921321256, 0.5617484009715226, 0.3837028125392342],
          [0.5634385611044553, 0.672472300417147, 0.011266430063146315],
          [0.16777673052080977, 0.9702676137000206, 0.601520491844329],
          [0.19107771092809633, 0.6209887142229743, 0.30432367394740356],
          [0.7609380362289568, 0.4107088700706961, 0.27852339673493376],
          [0.05601079191624403, 0.6172127735811344, 0.18390409957633136],
          [0.6853549103642731, 0.8787606847531632, 0.8793654972067946],
          [0.860890655078774, 0.8099778345487803, 0.7046097565540924],
          [0.3618545316497086, 0.2849730524798604, 0.9287668292910449],
          [0.4724530601738266, 0.7441801306881485, 0.46602764586955236],
          [0.9106598724399415, 0.892767328054341, 0.8595506820187084],
          [0.980559439370208, 0.4994549023655084, 0.27822796298844277],
          [0.6644572745252246, 0.9087606188518825, 0.9443300673106539],
          [0.31488672985813315, 0.37337261988083836, 0.9086545874709488],
          [0.9701422490706259, 0.34043972295464764, 0.27155986558754364],
          [0.6869203355984217, 0.8876115083859007, 0.4021378411101473],
          [0.5998671817528817, 0.5811238534690953, 0.7181133050757997],
          [0.16767555451487715, 0.7831119823135027, 0.6869132523360124],
          [0.9889357700156829, 0.7386964322321216, 0.680393892613994],
          [0.8306169793356317, 0.771278045289419, 0.9141436204884381],
          [0.40138564366912277, 0.6297156901329715, 0.8461961793794692],
          [0.8178675876069798, 0.78311248773471, 0.5624096339936324],
          [0.8304010664039991, 0.3958866875808472, 0.15647106116732357],
          [0.9877068427873675, 0.6725560957831684, 0.8348260938632719],
          [0.9756590631685597, 0.31964691847792825, 0.7600821697869036],
          [0.1338285705721014, 0.8386079951190131, 0.7031698904940803],
          [0.2590591708854534, 0.6087889280153227, 0.12353461838203339],
          [0.1795262991601443, 0.6204210383245907, 0.029718484105996756],
          [0.9692661419085732, 0.8236552726932775, 0.2975597206297578],
          [0.2717413603236529, 0.23023062624263824, 0.7439709963790858],
          [0.8441570067962137, 0.3863143749891652, 0.3027455549853948],
          [0.6932594963176144, 0.6628354070513242, 0.3153219462416158],
          [0.46958745800372403, 0.5382837674171578, 0.5813676560263662],
          [0.17359990348766974, 0.593929668165144, 0.9414890253072993],
          [0.26998440237594834, 0.17375516709288719, 0.7657895660637534],
          [0.09617148440581247, 0.4220384245159219, 0.1548226221424558],
          [0.5381794710842558, 0.30727561947165316, 0.18737709952336323],
          [0.42934678108316027, 0.1264539542932931, 0.6888204518936913],
          [0.6196628508334052, 0.6022983195376187, 0.49112167751341484],
          [0.31901364363168905, 0.29175188330978885, 0.4319902309992292],
          [0.90160931937047, 0.885659016082854, 0.9655887996128989],
          [0.292641157711897, 0.17886794522308347, 0.13384487296446157],
          [0.7837997233963436, 0.6294264836959282, 0.7850506094067871],
          [0.9513484783528485, 0.4432436036771492, 0.9271152514003411],
          [0.756817178647785, 0.5329711409742015, 0.6919367362756434],
          [0.4366264911133244, 0.5811285664719388, 0.5933941956717734]]
