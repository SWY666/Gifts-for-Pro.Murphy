if __name__ == "__main__":
    import matplotlib.pyplot as plt
    curve_list = [[0.13673975894338602, 0.13761700217070377, 0.14015047812136922, 0.1410271389401271, 0.13860839720048895, 0.13476135824143015, 0.13669975580099264, 0.13633976094504674, 0.13330639899221325, 0.13396873301683926, 0.1343747107945814, 0.13715864959934926, 0.137745233524729, 0.13723004794066807, 0.13294047900472006, 0.13843166684030095, 0.13738879651653046, 0.1385222612691749, 0.13338812384229726, 0.1336002332937185, 0.13229720850661347, 0.13594991686019958, 0.13549807152624002, 0.13314399566518118, 0.13574854912166176, 0.13416709711497216, 0.13179885978510764, 0.13369755564856933, 0.13172792310446585, 0.13196289251083612, 0.13310993279532615, 0.13551812049896322, 0.13541229168664914, 0.13365114020652677, 0.13249562910260557, 0.13251840112357355, 0.1340963977692789, 0.13156554614445956, 0.13269921455229988, 0.13449299723383665, 0.13278569156378098, 0.1325158131294275, 0.12840350124135763, 0.13089947555783035, 0.1281418276848624, 0.12683094748046794, 0.12684121963515074, 0.12707292097060557, 0.12799528253929462, 0.12727856793046743, 0.12730949703880778, 0.12569910070623633, 0.1289604642827815, 0.12479596639993748, 0.12634313617741127, 0.1253115282817558, 0.12561202737115335, 0.12286640902182352, 0.12380952162849894, 0.12390422549486868, 0.11977806315332826, 0.12278060841145064, 0.12123768184811712, 0.12279766663819855, 0.12166458968684032, 0.1204857634851791, 0.12068973015291544, 0.11998817458928993, 0.12169934278013865, 0.12045669098055305, 0.1199829987299312, 0.12068616564745378, 0.11811020982004147, 0.11949354428053202, 0.1195837645155231, 0.11812446605475209, 0.11779982244426157, 0.11712230517859297, 0.11677971048670166, 0.11653700625316439, 0.11541853064606407, 0.11642013168247481, 0.11630671065488656, 0.1173446694406208, 0.11666090985257824, 0.11503570908995482, 0.11465008561015651, 0.11375672258048738, 0.11500648878928468, 0.11467069677334189, 0.11411275013764813, 0.11443744509959611, 0.11509710658576733, 0.11405921993875576, 0.11490730585566648, 0.11639291050015912, 0.11359724930105515, 0.1129599230486775, 0.11186868287629936, 0.11173612151938654, 0.11061722654177049, 0.10923249922880778, 0.1119119541332169, 0.10957711360173791, 0.11035636090718697, 0.11050216406196375, 0.11059722379400264, 0.10943549970455724, 0.10956271955621899, 0.1094120841559086, 0.10801775756165019, 0.10857134540797848, 0.1082743015849501, 0.10965573870738761, 0.11055405664595641, 0.10777982148132406, 0.10821761779494014, 0.10785553191079096, 0.10625310255063813, 0.1058961762670583, 0.10713132863929883, 0.10756408294045064, 0.10697127167679202, 0.10568748763407616, 0.10398880180987499, 0.1055870269008545, 0.10605664833380987, 0.10561839322433823, 0.10527512800819275, 0.1058177135294438, 0.10534889366919434, 0.10504329235845851, 0.1050571580827206, 0.10477175552149161, 0.10589304375610767, 0.10320530717096377, 0.10519364727428782, 0.1057029511536854, 0.104420831924219, 0.10545802487389706, 0.10309362726187973, 0.10259440678885032, 0.10271032444487027, 0.10499346669827259, 0.10325424853233825, 0.10417175270083612, 0.10229725370667542, 0.101959683474323, 0.10163449526169592, 0.10211617559380042, 0.10141878610238235, 0.10023775564141121, 0.09924396045811502, 0.10094119237057511, 0.10072784414662499, 0.10056059094941125, 0.09858931221307037, 0.09842510129303308, 0.09936452246873687, 0.0985144971412395, 0.09901132627103991, 0.09846644773230934, 0.09840098613899106, 0.09849175772276209, 0.09873903443624474, 0.09774996658982142, 0.09717613722850944, 0.09663884297157405, 0.09690894679161345, 0.09673555696746902, 0.09906624200488338, 0.09797089446471428, 0.09822489749948635, 0.09713814766674055, 0.09512770963180776, 0.09558203449941857], [0.14219245195487396, 0.08086500556669432, 0.07940551883108982, 0.07850629097119907, 0.07850604671782878, 0.0777190179058687, 0.0773246901902479, 0.07751289206762028, 0.07716386055096533, 0.07805018399030936, 0.07767539202465232, 0.07775933000447846, 0.07739942926483875, 0.07721125256100052, 0.07810242176838969, 0.0773120007452915, 0.07804245352350682, 0.07708488938191649, 0.07852463740684876, 0.0779630599799748, 0.07790960647653676, 0.0766879929915956, 0.07735363524077363, 0.07665221132129987, 0.07598709197821586, 0.07718405737076052, 0.07653650486786608, 0.07615500335110137, 0.07721485370342963, 0.07630298410773267, 0.07630824802328992, 0.07630390861838497, 0.0762091897477437, 0.07583035484803731, 0.07689658590920957, 0.07519908026122742, 0.07619073504083193, 0.07626562871613393, 0.07554532024993435, 0.07558498856154253, 0.0757416875454291, 0.07622875512312186, 0.07590896640233982, 0.07581663927355778, 0.07539984631592858, 0.07532196791668593, 0.0753063426855935, 0.07488400372271796, 0.0753736786452164, 0.07457488716719433, 0.07411926156097899, 0.0743513983508855, 0.07504971715905094, 0.07516479033195955, 0.07517910599370274, 0.07552502573384978, 0.07462807065486292, 0.07392762051003414, 0.07508062949348948, 0.07575790587063445, 0.07435880411169958, 0.07416961536006775, 0.07354468096105567, 0.07292536855296008, 0.073831667746531, 0.07398710873050263, 0.07390163194408719, 0.0734217316261187, 0.07438698432460715, 0.0741677905671728, 0.07355630588353047, 0.07308002012528415, 0.07336482707114758, 0.07307348001605464, 0.07309852307339548, 0.07353897562662605, 0.07337164086815914, 0.07293273160369221, 0.07311159988609979, 0.07313286333239917, 0.07284746400675736, 0.07312892831265547, 0.07300633797029411, 0.07303915511955095, 0.07280617420085031, 0.07226615918587184, 0.07324786071144854, 0.07231946461912134, 0.07302426094373424, 0.07252050821450315, 0.07248389183383193, 0.07239148655905409, 0.07233071531064812, 0.07176662014885453, 0.0719378765768913, 0.07206747488735714, 0.07144666445635976, 0.0715922866718188, 0.07151629210072688, 0.0720559292133104, 0.07150287213802836, 0.0711919227380961, 0.07094561311058718, 0.07078178947080146, 0.07058436366145746, 0.07092743541636397, 0.06967865493883665, 0.07069953080904308, 0.06954107746867018, 0.0702742226986439, 0.07038698283000683, 0.06996673941166755, 0.07015354181906197, 0.07001410387930095, 0.06984482303146375, 0.06958180106209506, 0.06928867569799632, 0.06958914580230358, 0.06962332269729518, 0.0694274116844874, 0.06998166650841228, 0.06979759951051366, 0.07040465692077627, 0.06928694198584454, 0.07015642018714548, 0.0697180871235168, 0.0698674233920352, 0.06879544368305672, 0.06939587579391177, 0.06897420793238603, 0.06858580235813709, 0.06853895365361398, 0.06847376340121376, 0.06817629704559716, 0.06841140841442857, 0.06892428984223056, 0.06870806162970168, 0.06935738549976575, 0.06828978190307879, 0.06845936610759926, 0.06826344994508456, 0.06840188442278886, 0.06791419076589235, 0.0685216898543687, 0.06866569730055513, 0.0676911675506854, 0.06804655334791965, 0.06880567826477889, 0.06874377021363079, 0.06815949470007668, 0.06853370458495799, 0.06857820422535137, 0.06877661148388971, 0.06829817093312816, 0.06786585113094538, 0.06803766961177153, 0.06768197204496218, 0.06777992262338131, 0.06784022197469748, 0.06780048657456408, 0.06753105597320745, 0.06765393106546114, 0.06837349466837131, 0.06829780346334453, 0.06718970793256303, 0.06782124826058246, 0.06740765165235242, 0.06757516405728939, 0.06739484238685925, 0.06711653762913342, 0.06693792042490705, 0.0666465390418834, 0.06640885689243856, 0.06672951221495378, 0.06689823411099631, 0.06713059949470483], [0.13646018162959297, 0.12415826965611398, 0.1230522025498046, 0.12427812873609193, 0.12344030085233351, 0.12343863041381564, 0.12514574452428834, 0.12398731265382248, 0.12544541324912867, 0.12145834354610559, 0.12357260039188182, 0.12472194100348843, 0.12308491961850683, 0.12419861512488391, 0.1230735284766024, 0.1250120265615126, 0.12449316134528572, 0.12616556200268542, 0.12537524128081828, 0.12360929493290704, 0.12262913158982877, 0.12451339800496271, 0.1235720116160268, 0.12409386077590231, 0.12413652782491809, 0.12210692019054097, 0.12327776765631984, 0.12086441039121902, 0.12354635903101734, 0.12158704713663075, 0.12089635341007932, 0.12296415090250053, 0.12080678628566122, 0.12338341663644745, 0.12151805443125316, 0.12085843354566284, 0.12084114731804832, 0.11973085436807614, 0.11880184960780252, 0.11916360931725156, 0.11942408068146322, 0.11925134614404306, 0.1196774850868918, 0.11638176915774316, 0.11728335100556406, 0.11742443682560191, 0.11553020811722847, 0.11479850790574528, 0.11412141484672374, 0.11223272110541964, 0.11344981091043067, 0.1141434193935084, 0.11297175892545905, 0.11154687908418066, 0.1114784982234811, 0.11189358336537183, 0.11087341631830724, 0.11051985705711713, 0.1088691458542789, 0.11007645887391333, 0.11053304943824634, 0.10947019535640387, 0.10975507082516282, 0.11079512340662186, 0.10921346898585926, 0.11037567035576261, 0.10812218615767229, 0.10924627541275, 0.10733032689534086, 0.10764855450266439, 0.10822902326154411, 0.10683055827174162, 0.10529811128727314, 0.10603692238395693, 0.10558921947571849, 0.10501870532073794, 0.10570395279589757, 0.10365329265970431, 0.1030051589146502, 0.10303214310158508, 0.10261898137468643, 0.10248189658188026, 0.10282129263189968, 0.10234043192693487, 0.10208705855755149, 0.10153036979486871, 0.10111278628610347, 0.10050331689248584, 0.10112867188428469, 0.1010428525184958, 0.10205742138528835, 0.10055332023968803, 0.10192163918993802, 0.10010409362195483, 0.10027589372435557, 0.10091929397493489, 0.10014930741341335, 0.10053855426679777, 0.09943858500083141, 0.10021228006607405, 0.09963926700636029, 0.09884400591140177, 0.09829310844498737, 0.09895294499547533, 0.09893419714579202, 0.09805611050028099, 0.09695273471841596, 0.09805814006588236, 0.0973080310014144, 0.09682454093988341, 0.0965682233909895, 0.09501601744275683, 0.09543109847573163, 0.09536806701422954, 0.09541147981601891, 0.09433509069816073, 0.09407948279450526, 0.09431451442113394, 0.09444330126359349, 0.0935166707351959, 0.09310643296209611, 0.09344932593960714, 0.09303815869191596, 0.09217994792672793, 0.09188994379351681, 0.09291555503380884, 0.09286899159087761, 0.0921507238584228, 0.09318633820756354, 0.09204634839491807, 0.09139353908324789, 0.09097042557324424, 0.0918064340108582, 0.09140909666739812, 0.0904499177739438, 0.09026942675567488, 0.08939613709128677, 0.09007052756281196, 0.08898354248101102, 0.08900469743626944, 0.0891058367235084, 0.08938664850667752, 0.0884709708226229, 0.0885705036114769, 0.08830643598896952, 0.08786835214086029, 0.08811004140091597, 0.08762023679975105, 0.08707688081869117, 0.08694373386815388, 0.08699783809669855, 0.08672164676031144, 0.08637998069300998, 0.08678475726331407, 0.08655129636194539, 0.08674226172954674, 0.08543943155113445, 0.08624263309874211, 0.08558948577644329, 0.08597186958302888, 0.08535246193149422, 0.08580186169877993, 0.084535884599677, 0.08498478114057614, 0.08473359148387971, 0.08375766818732878, 0.08383564878051628, 0.08283460274112657, 0.08285580720671376, 0.08270647461576733, 0.08232574805026102, 0.08308482705528256, 0.08295139439638285, 0.08274844628632036, 0.08271526062342016, 0.08263307675235923], [0.1415548945984438, 0.13750314333223163, 0.13486700479751051, 0.13444058439115597, 0.13293636603107564, 0.1341421328439102, 0.13660915139656354, 0.13400044281651521, 0.13194740927055829, 0.13342124977168432, 0.13095513050449212, 0.1358943043636318, 0.13512110562216595, 0.13432637208166753, 0.13191929015855672, 0.1305065849388382, 0.13467996735117435, 0.13315677126151892, 0.13190399959241703, 0.13413056362960607, 0.13411547729752993, 0.13066735264119958, 0.1296367503834464, 0.1291209380627432, 0.127605728090115, 0.12821617450625575, 0.128212753624719, 0.12755549367577523, 0.12755491175022635, 0.1265655648807211, 0.1254543699543771, 0.12852471705018748, 0.1287435732915541, 0.12925756284353884, 0.12807397264564443, 0.1257459183112521, 0.1263993041386119, 0.1263424290767647, 0.12621119634423056, 0.12697021417552362, 0.12562681035441386, 0.12444587390166964, 0.1263198410239102, 0.12665231975291544, 0.12433459852202522, 0.12374938119080306, 0.12491145876684188, 0.1233732338141313, 0.12328674321975157, 0.1230482905064559, 0.12319166527867909, 0.12079564010824632, 0.11845158957124527, 0.12162995364032196, 0.11911341753243698, 0.12020412109041227, 0.1199180727828288, 0.11924750153285239, 0.11822860970975894, 0.11786624927602415, 0.11863562669238972, 0.12020719100060662, 0.11800808625694696, 0.11752588206012815, 0.1180029219834706, 0.11819471342534138, 0.11765247176693643, 0.11883001469898583, 0.11735578173831829, 0.1165425351663083, 0.1175095096934186, 0.11655091747430987, 0.11602771662341702, 0.11516650529129306, 0.11469812897739182, 0.11441929189465545, 0.1132082088297521, 0.1154027407419769, 0.11488196255108034, 0.11374633715070796, 0.11452309299799265, 0.11371771866577784, 0.1123909975619916, 0.11292688782354834, 0.11257269620827941, 0.11143126101077679, 0.11090915028534035, 0.10940570257802573, 0.10867806893336818, 0.11027460412284454, 0.10959303201711346, 0.11039839755977468, 0.10945745482722269, 0.10985594055517489, 0.10900587840652523, 0.10887303570818645, 0.10730434460925209, 0.10815669699596428, 0.10703700258599422, 0.10821015759169801, 0.10715723811609978, 0.10875896699887973, 0.10640940384286608, 0.10731390299325631, 0.1072957603514958, 0.10539834468976417, 0.10534812455415546, 0.1047174808124081, 0.10492195759291598, 0.10442041496860714, 0.10565749696825158, 0.1036234095928125, 0.10310993424543854, 0.10459229498260136, 0.10490095244379201, 0.10162036000927625, 0.10168143321667227, 0.1023347363850688, 0.10345397849925526, 0.10352309681607667, 0.10311206694719748, 0.10273157207425683, 0.10122565120912445, 0.1009048521755688, 0.10104906355784979, 0.10050253229607879, 0.0995147886187957, 0.09954442541755462, 0.09940059729653623, 0.09976530190030777, 0.09954418035832091, 0.09819112396272742, 0.09783813925048267, 0.09845545910762817, 0.0982910941156649, 0.09714636953313183, 0.09838568559385243, 0.09765547567625263, 0.09549762434243855, 0.09716350502502993, 0.09642232777039336, 0.0959304666685646, 0.09570063376093277, 0.09583487289440441, 0.0963042029242731, 0.09501442941145062, 0.09614383693431985, 0.0954302990360646, 0.09626513741394538, 0.09576282528648897, 0.0965420125556145, 0.09547665456338761, 0.09540630674481197, 0.09499031791552888, 0.09431915997799842, 0.09429321788006617, 0.09438863586460977, 0.09507369333555618, 0.09445266212789757, 0.09365850769428624, 0.09309408423120641, 0.09362450857260031, 0.09240769180674999, 0.09287907867845852, 0.09317372152596691, 0.09323372061943802, 0.09318247999208454, 0.09289256263721113, 0.09348055055337501, 0.09264254221841911, 0.09260869393180725, 0.09313489867979989, 0.09250555915595642, 0.09215440738916911, 0.0926097460328372, 0.09183086836271166], [0.13506903306483614, 0.11735356719699316, 0.11952727032130357, 0.11863016501097531, 0.1160229460292999, 0.1161267066008876, 0.11720274062248322, 0.11552911740650681, 0.11400585738417804, 0.11459576169726575, 0.11246755053314812, 0.11345466490800789, 0.11318768712998636, 0.11322280194921377, 0.1120014936229643, 0.11245651177117857, 0.11232062437992858, 0.1126023722931208, 0.11158568379839337, 0.11035177313331776, 0.10829330547407563, 0.11126722999687395, 0.10896322146023477, 0.10859027159622267, 0.10776319578497165, 0.1082745805367894, 0.10943486076639179, 0.10912091306563498, 0.10915084308716753, 0.10744086064037657, 0.10779799694958876, 0.1096853194733792, 0.10751302560907613, 0.10724107906960294, 0.10728924939058773, 0.10570909725347111, 0.10867440225255147, 0.10826757929496744, 0.10720001029790283, 0.10827428092501942, 0.10741697020565075, 0.10884047488638131, 0.10831043637611397, 0.10836010128121956, 0.10793792931174526, 0.10733051410668489, 0.10779904416389288, 0.10654962337395746, 0.10683892190096585, 0.10583596767565598, 0.1056725907065793, 0.10620109205740391, 0.10501705342276942, 0.10514892051479467, 0.10639214587462711, 0.10648312914188655, 0.10643589266382616, 0.10553545467271638, 0.10526838331151156, 0.10589634823173319, 0.10385780576961764, 0.10514201365080778, 0.10462465046579254, 0.10441030614952204, 0.1031659951937353, 0.10334146264108719, 0.103617534341031, 0.10334334145576207, 0.10293880778414286, 0.10362245980438078, 0.10295220642708403, 0.10143682355835715, 0.10126715945314602, 0.10143859752438708, 0.10136111153155672, 0.10059037780664075, 0.1002894352133871, 0.10062787100124737, 0.09899847859671533, 0.09935195837556302, 0.09877198838253151, 0.09890731659998897, 0.09827423441610869, 0.0982705140822206, 0.0978046785081623, 0.09827005682559192, 0.09791166799913656, 0.09772091456023742, 0.09807792252009297, 0.09759598496134034, 0.09698613145849475, 0.09731793592050053, 0.09736335628140756, 0.09687628127620537, 0.09709496297591125, 0.09612576218366176, 0.09549152684265702, 0.09540731092309034, 0.09544558666724423, 0.09539670027257405, 0.09476860083775421, 0.09475716754396639, 0.0953612632507458, 0.09411168101054622, 0.09476471701565284, 0.09502541600306251, 0.09413531241299686, 0.09349144854378992, 0.0940869748540998, 0.09306048137308193, 0.09223471903245326, 0.09263265029534348, 0.092269589761958, 0.09189424602851681, 0.09219346256531355, 0.09183635317691177, 0.09334287611426734, 0.0913638527016791, 0.09178089368117909, 0.09119305156482196, 0.09090563735823687, 0.09049040025132511, 0.090400198365771, 0.09043407159121375, 0.09024023178833299, 0.08977570300564601, 0.09005812597172848, 0.0900862659440961, 0.08994242307553625, 0.09012534766244329, 0.0892851384541996, 0.08863966154708876, 0.08898965698879517, 0.08861102684539443, 0.08913098732984245, 0.08873150582909821, 0.08922982595344066, 0.08838329647418801, 0.08814545722850055, 0.08781666918200788, 0.08792389956594224, 0.08884996407009298, 0.08851838753404309, 0.08823675875580515, 0.08733848004922007, 0.08709725197284665, 0.08678936902163341, 0.08597715929196273, 0.08627966781203099, 0.0852460591078766, 0.08581824750298529, 0.08451790284268068, 0.08517044791209297, 0.08492083048615703, 0.08520170430957667, 0.08480343213306354, 0.08673942542650209, 0.08612532528292963, 0.08536450473092701, 0.08541529761569906, 0.08457424422968908, 0.08378875022693069, 0.08454967253787868, 0.08441702426228047, 0.08427579639107145, 0.08407174775356355, 0.08441852748293119, 0.0846423326262479, 0.08379460595521113, 0.08347257485808929, 0.08356505064879831, 0.08278459759203888, 0.08307218248791493, 0.08324470914954098, 0.08258870669278257, 0.08249159397286765], [0.1372436942516901, 0.13229898103987187, 0.1318350643117306, 0.12624175718551942, 0.130173088257395, 0.12625981521964863, 0.13077042869588967, 0.13046575929611975, 0.12808634942529726, 0.1270289803777826, 0.12650633466467492, 0.12359701280943854, 0.12298611349216859, 0.12248888153788236, 0.11862832865847743, 0.11771957011240704, 0.1144586642983934, 0.11564907539923316, 0.11481634339131723, 0.11290157577928572, 0.1135882978869601, 0.11511572146661764, 0.11220979319681881, 0.11074820238085711, 0.1106770193213062, 0.11008284625559242, 0.10905528595757197, 0.10799172612871744, 0.10850164261334874, 0.10786066736924578, 0.10745528466630934, 0.10625799741489339, 0.1059920553631066, 0.10392471780838078, 0.10735309242049107, 0.1058625918893624, 0.10536901262999107, 0.10615041281580462, 0.10409815131164916, 0.1053530032841397, 0.10539174659461292, 0.10547868734212396, 0.10473769953588656, 0.10454996301157142, 0.10346154652599107, 0.10401957683407141, 0.10419305779199525, 0.10366330900032825, 0.10302038261183928, 0.10329674701497427, 0.10201808701636658, 0.1031717236706355, 0.10270238245479042, 0.10255090961066333, 0.10285567459810714, 0.10288180096879358, 0.10241223817583875, 0.1020393135540898, 0.10195982772000367, 0.10080818631147795, 0.10118454372234557, 0.1009235617883025, 0.10057729221144589, 0.10047358858083297, 0.09942243595705515, 0.09945223919635557, 0.09846697310346744, 0.09854250506709086, 0.09931920991145378, 0.09873426713323213, 0.09818626873402259, 0.09855186007503308, 0.09794253981310555, 0.09749287618715335, 0.09721751181990285, 0.09671287069252572, 0.09582399488260608, 0.09585663350901523, 0.09625660621375316, 0.09554678478883456, 0.09498289799807773, 0.0961201514471649, 0.0954775367076481, 0.09583047127328517, 0.09432103874871428, 0.09461160052576993, 0.09469820264695011, 0.09518398391675105, 0.09397241138022269, 0.09410828424275892, 0.09408572876284453, 0.09386597973824842, 0.09438743489833457, 0.094231728797073, 0.09404887992619433, 0.0940268922765625, 0.09348670559684716, 0.09285474574215599, 0.09258145958491963, 0.09304681459088446, 0.09317465681600053, 0.09263670601425472, 0.09386999017487027, 0.09297934362020431, 0.09329746743122426, 0.09213103665813185, 0.09259344738142805, 0.09194342985772744, 0.09131558138250526, 0.09155674669099055, 0.09158939595826207, 0.09167654196104832, 0.09179002881044328, 0.09151432669842281, 0.09060212712450473, 0.09142368408654516, 0.09072712849531882, 0.09066929954492489, 0.09042075266405936, 0.08995453357770798, 0.09027673169937973, 0.09004299736037867, 0.08978275283220957, 0.0896874657581124, 0.08932057435844956, 0.08914375314932668, 0.08944166936709505, 0.08948889396159035, 0.08975291302901944, 0.08916091762383035, 0.08962868702455513, 0.08887657715934401, 0.08897022044015442, 0.0885998262735646, 0.08814437153977311, 0.08789567193526418, 0.0877845134400436, 0.0885668391159522, 0.08784595527753414, 0.08742194258417385, 0.08783463002655671, 0.08763268580779464, 0.0871897317924396, 0.08762407082757143, 0.08638690961233508, 0.08634424418483823, 0.08681373861401523, 0.08699522827971692, 0.08711018394916019, 0.08642196078914496, 0.0858804903025693, 0.08595877587606145, 0.08511804285906933, 0.08525861248168015, 0.08545629803903729, 0.08568192091703307, 0.08471810899357299, 0.08533332679473687, 0.0846859143855914, 0.08523599303420062, 0.08528492050050421, 0.08501382741993749, 0.08418376853282247, 0.08466534465664444, 0.08467884296662395, 0.08495657687033718, 0.08400505703435664, 0.08399378779107246, 0.08374942268736871, 0.08341812399017909, 0.08345602233431985, 0.08326803369008087, 0.0828789744005163, 0.08402021996758405, 0.0835033153957232, 0.083405994769698]]
    m = curve_list[0]
    m1 = curve_list[1]
    m2 = curve_list[2]
    m3 = curve_list[3]
    m4 = curve_list[4]
    m5 = curve_list[5]

    plt.figure()
    plt.plot([j for j in range(len(m))], m, "b", marker="|")
    plt.plot([j for j in range(len(m1))], m1, "b--")
     # plt.fill_between([j for j in range(len(m1))], m1 - s1, m1 + s1, color='blue', alpha=0.2)
    plt.plot([j for j in range(len(m2))], m2, "b")
      # plt.fill_between([j for j in range(len(m2))], m2 - s2, m2 + s2, color='yellow', alpha=0.2)
    plt.plot([j for j in range(len(m3))], m3, "b-.")
    # plt.fill_between([j for j in range(len(m3))], m3 - s3, m3 + s3, color='red', alpha=0.2)
    plt.plot([j for j in range(len(m4))], m4, "b--.")
     # plt.fill_between([j for j in range(len(m4))], m4 - s4, m4 + s4, color='black', alpha=0.2)
    plt.plot([j for j in range(len(m5))], m5, "b:")
    # plt.fill_between([j for j in range(len(m5))], m5 - s5, m5 + s5, color='yellow', alpha=0.2)
    plt.legend(
        ["SC", "AM_RAW", "AM-per-5 epochs", "Clustering-per 5 epochs", "AM & most uncertainty", "C & most uncertainty"])
    plt.xlabel("num-of-epochs")
    plt.ylabel("MSE")
    plt.title("cross-validation")
    plt.show()
