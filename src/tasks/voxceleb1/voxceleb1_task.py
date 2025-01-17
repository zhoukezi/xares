from xares.task_base import TaskBase, TaskConfig


class VoxCeleb1Task(TaskBase):
    def __init__(self, encoder):
        data_key = "speakerid"
        self.class_label_maps = {
            "id10003": 0,
            "id10004": 1,
            "id10005": 2,
            "id10006": 3,
            "id10007": 4,
            "id10008": 5,
            "id10009": 6,
            "id10010": 7,
            "id10011": 8,
            "id10012": 9,
            "id10013": 10,
            "id10014": 11,
            "id10015": 12,
            "id10016": 13,
            "id10017": 14,
            "id10001": 15,
            "id10018": 16,
            "id10019": 17,
            "id10020": 18,
            "id10021": 19,
            "id10022": 20,
            "id10023": 21,
            "id10024": 22,
            "id10025": 23,
            "id10030": 24,
            "id10031": 25,
            "id10032": 26,
            "id10033": 27,
            "id10026": 28,
            "id10027": 29,
            "id10028": 30,
            "id10029": 31,
            "id10034": 32,
            "id10035": 33,
            "id10036": 34,
            "id10037": 35,
            "id10038": 36,
            "id10039": 37,
            "id10040": 38,
            "id10041": 39,
            "id10042": 40,
            "id10043": 41,
            "id10044": 42,
            "id10045": 43,
            "id10046": 44,
            "id10047": 45,
            "id10048": 46,
            "id10050": 47,
            "id10049": 48,
            "id10051": 49,
            "id10052": 50,
            "id10053": 51,
            "id10054": 52,
            "id10055": 53,
            "id10056": 54,
            "id10057": 55,
            "id10058": 56,
            "id10060": 57,
            "id10059": 58,
            "id10061": 59,
            "id10062": 60,
            "id10063": 61,
            "id10064": 62,
            "id10065": 63,
            "id10067": 64,
            "id10066": 65,
            "id10068": 66,
            "id10069": 67,
            "id10070": 68,
            "id10002": 69,
            "id10071": 70,
            "id10072": 71,
            "id10073": 72,
            "id10074": 73,
            "id10075": 74,
            "id10076": 75,
            "id10078": 76,
            "id10079": 77,
            "id10080": 78,
            "id10081": 79,
            "id10082": 80,
            "id10083": 81,
            "id10084": 82,
            "id10085": 83,
            "id10087": 84,
            "id10086": 85,
            "id10088": 86,
            "id10091": 87,
            "id10089": 88,
            "id10090": 89,
            "id10092": 90,
            "id10077": 91,
            "id10093": 92,
            "id10094": 93,
            "id10095": 94,
            "id10096": 95,
            "id10097": 96,
            "id10098": 97,
            "id10099": 98,
            "id10101": 99,
            "id10100": 100,
            "id10102": 101,
            "id10103": 102,
            "id10104": 103,
            "id10105": 104,
            "id10106": 105,
            "id10107": 106,
            "id10108": 107,
            "id10109": 108,
            "id10110": 109,
            "id10111": 110,
            "id10112": 111,
            "id10113": 112,
            "id10114": 113,
            "id10115": 114,
            "id10116": 115,
            "id10118": 116,
            "id10119": 117,
            "id10120": 118,
            "id10121": 119,
            "id10122": 120,
            "id10123": 121,
            "id10124": 122,
            "id10125": 123,
            "id10126": 124,
            "id10127": 125,
            "id10128": 126,
            "id10129": 127,
            "id10130": 128,
            "id10131": 129,
            "id10132": 130,
            "id10133": 131,
            "id10134": 132,
            "id10135": 133,
            "id10136": 134,
            "id10117": 135,
            "id10137": 136,
            "id10138": 137,
            "id10139": 138,
            "id10140": 139,
            "id10141": 140,
            "id10142": 141,
            "id10143": 142,
            "id10144": 143,
            "id10145": 144,
            "id10146": 145,
            "id10147": 146,
            "id10148": 147,
            "id10149": 148,
            "id10150": 149,
            "id10151": 150,
            "id10152": 151,
            "id10153": 152,
            "id10154": 153,
            "id10155": 154,
            "id10156": 155,
            "id10157": 156,
            "id10158": 157,
            "id10159": 158,
            "id10160": 159,
            "id10161": 160,
            "id10162": 161,
            "id10163": 162,
            "id10164": 163,
            "id10165": 164,
            "id10166": 165,
            "id10167": 166,
            "id10168": 167,
            "id10169": 168,
            "id10170": 169,
            "id10171": 170,
            "id10172": 171,
            "id10173": 172,
            "id10174": 173,
            "id10175": 174,
            "id10176": 175,
            "id10177": 176,
            "id10178": 177,
            "id10179": 178,
            "id10180": 179,
            "id10181": 180,
            "id10182": 181,
            "id10183": 182,
            "id10184": 183,
            "id10188": 184,
            "id10185": 185,
            "id10189": 186,
            "id10186": 187,
            "id10190": 188,
            "id10191": 189,
            "id10192": 190,
            "id10193": 191,
            "id10195": 192,
            "id10196": 193,
            "id10194": 194,
            "id10197": 195,
            "id10198": 196,
            "id10199": 197,
            "id10187": 198,
            "id10200": 199,
            "id10201": 200,
            "id10202": 201,
            "id10203": 202,
            "id10204": 203,
            "id10205": 204,
            "id10206": 205,
            "id10207": 206,
            "id10208": 207,
            "id10209": 208,
            "id10210": 209,
            "id10211": 210,
            "id10212": 211,
            "id10213": 212,
            "id10214": 213,
            "id10215": 214,
            "id10216": 215,
            "id10217": 216,
            "id10218": 217,
            "id10219": 218,
            "id10220": 219,
            "id10221": 220,
            "id10222": 221,
            "id10223": 222,
            "id10224": 223,
            "id10225": 224,
            "id10226": 225,
            "id10227": 226,
            "id10228": 227,
            "id10229": 228,
            "id10230": 229,
            "id10231": 230,
            "id10232": 231,
            "id10234": 232,
            "id10233": 233,
            "id10235": 234,
            "id10236": 235,
            "id10237": 236,
            "id10238": 237,
            "id10239": 238,
            "id10240": 239,
            "id10241": 240,
            "id10242": 241,
            "id10243": 242,
            "id10244": 243,
            "id10245": 244,
            "id10246": 245,
            "id10247": 246,
            "id10250": 247,
            "id10251": 248,
            "id10249": 249,
            "id10252": 250,
            "id10253": 251,
            "id10254": 252,
            "id10248": 253,
            "id10255": 254,
            "id10256": 255,
            "id10257": 256,
            "id10258": 257,
            "id10259": 258,
            "id10261": 259,
            "id10260": 260,
            "id10262": 261,
            "id10263": 262,
            "id10264": 263,
            "id10265": 264,
            "id10266": 265,
            "id10267": 266,
            "id10268": 267,
            "id10269": 268,
            "id10270": 269,
            "id10272": 270,
            "id10273": 271,
            "id10274": 272,
            "id10275": 273,
            "id10276": 274,
            "id10277": 275,
            "id10278": 276,
            "id10271": 277,
            "id10279": 278,
            "id10280": 279,
            "id10281": 280,
            "id10282": 281,
            "id10283": 282,
            "id10285": 283,
            "id10284": 284,
            "id10286": 285,
            "id10287": 286,
            "id10288": 287,
            "id10289": 288,
            "id10290": 289,
            "id10291": 290,
            "id10292": 291,
            "id10293": 292,
            "id10294": 293,
            "id10295": 294,
            "id10296": 295,
            "id10297": 296,
            "id10298": 297,
            "id10299": 298,
            "id10300": 299,
            "id10301": 300,
            "id10302": 301,
            "id10303": 302,
            "id10304": 303,
            "id10305": 304,
            "id10306": 305,
            "id10307": 306,
            "id10308": 307,
            "id10309": 308,
            "id10310": 309,
            "id10311": 310,
            "id10312": 311,
            "id10313": 312,
            "id10314": 313,
            "id10315": 314,
            "id10316": 315,
            "id10317": 316,
            "id10318": 317,
            "id10319": 318,
            "id10320": 319,
            "id10322": 320,
            "id10321": 321,
            "id10323": 322,
            "id10324": 323,
            "id10325": 324,
            "id10326": 325,
            "id10327": 326,
            "id10328": 327,
            "id10329": 328,
            "id10330": 329,
            "id10331": 330,
            "id10332": 331,
            "id10333": 332,
            "id10334": 333,
            "id10335": 334,
            "id10336": 335,
            "id10337": 336,
            "id10338": 337,
            "id10339": 338,
            "id10340": 339,
            "id10341": 340,
            "id10342": 341,
            "id10343": 342,
            "id10344": 343,
            "id10345": 344,
            "id10346": 345,
            "id10347": 346,
            "id10348": 347,
            "id10349": 348,
            "id10350": 349,
            "id10351": 350,
            "id10352": 351,
            "id10353": 352,
            "id10354": 353,
            "id10355": 354,
            "id10356": 355,
            "id10357": 356,
            "id10358": 357,
            "id10359": 358,
            "id10360": 359,
            "id10361": 360,
            "id10362": 361,
            "id10363": 362,
            "id10364": 363,
            "id10365": 364,
            "id10366": 365,
            "id10367": 366,
            "id10368": 367,
            "id10369": 368,
            "id10370": 369,
            "id10371": 370,
            "id10372": 371,
            "id10373": 372,
            "id10374": 373,
            "id10375": 374,
            "id10376": 375,
            "id10377": 376,
            "id10378": 377,
            "id10379": 378,
            "id10380": 379,
            "id10381": 380,
            "id10382": 381,
            "id10383": 382,
            "id10384": 383,
            "id10385": 384,
            "id10386": 385,
            "id10387": 386,
            "id10388": 387,
            "id10389": 388,
            "id10390": 389,
            "id10391": 390,
            "id10392": 391,
            "id10393": 392,
            "id10394": 393,
            "id10395": 394,
            "id10396": 395,
            "id10399": 396,
            "id10400": 397,
            "id10401": 398,
            "id10402": 399,
            "id10403": 400,
            "id10404": 401,
            "id10405": 402,
            "id10406": 403,
            "id10407": 404,
            "id10408": 405,
            "id10409": 406,
            "id10410": 407,
            "id10411": 408,
            "id10412": 409,
            "id10413": 410,
            "id10414": 411,
            "id10415": 412,
            "id10416": 413,
            "id10417": 414,
            "id10418": 415,
            "id10419": 416,
            "id10420": 417,
            "id10421": 418,
            "id10422": 419,
            "id10423": 420,
            "id10424": 421,
            "id10425": 422,
            "id10426": 423,
            "id10427": 424,
            "id10428": 425,
            "id10430": 426,
            "id10431": 427,
            "id10429": 428,
            "id10432": 429,
            "id10434": 430,
            "id10433": 431,
            "id10435": 432,
            "id10436": 433,
            "id10437": 434,
            "id10438": 435,
            "id10439": 436,
            "id10440": 437,
            "id10441": 438,
            "id10442": 439,
            "id10443": 440,
            "id10444": 441,
            "id10445": 442,
            "id10446": 443,
            "id10447": 444,
            "id10448": 445,
            "id10449": 446,
            "id10450": 447,
            "id10451": 448,
            "id10452": 449,
            "id10453": 450,
            "id10454": 451,
            "id10459": 452,
            "id10455": 453,
            "id10460": 454,
            "id10456": 455,
            "id10457": 456,
            "id10458": 457,
            "id10461": 458,
            "id10462": 459,
            "id10463": 460,
            "id10464": 461,
            "id10465": 462,
            "id10466": 463,
            "id10467": 464,
            "id10469": 465,
            "id10468": 466,
            "id10470": 467,
            "id10471": 468,
            "id10472": 469,
            "id10473": 470,
            "id10474": 471,
            "id10475": 472,
            "id10476": 473,
            "id10477": 474,
            "id10478": 475,
            "id10479": 476,
            "id10480": 477,
            "id10481": 478,
            "id10482": 479,
            "id10483": 480,
            "id10484": 481,
            "id10485": 482,
            "id10486": 483,
            "id10487": 484,
            "id10488": 485,
            "id10489": 486,
            "id10490": 487,
            "id10491": 488,
            "id10492": 489,
            "id10493": 490,
            "id10494": 491,
            "id10495": 492,
            "id10496": 493,
            "id10497": 494,
            "id10498": 495,
            "id10500": 496,
            "id10501": 497,
            "id10499": 498,
            "id10502": 499,
            "id10503": 500,
            "id10504": 501,
            "id10506": 502,
            "id10505": 503,
            "id10397": 504,
            "id10398": 505,
            "id10507": 506,
            "id10508": 507,
            "id10509": 508,
            "id10511": 509,
            "id10512": 510,
            "id10513": 511,
            "id10510": 512,
            "id10514": 513,
            "id10515": 514,
            "id10516": 515,
            "id10517": 516,
            "id10518": 517,
            "id10519": 518,
            "id10520": 519,
            "id10546": 520,
            "id10521": 521,
            "id10522": 522,
            "id10523": 523,
            "id10524": 524,
            "id10525": 525,
            "id10526": 526,
            "id10527": 527,
            "id10528": 528,
            "id10529": 529,
            "id10530": 530,
            "id10531": 531,
            "id10532": 532,
            "id10533": 533,
            "id10534": 534,
            "id10535": 535,
            "id10536": 536,
            "id10537": 537,
            "id10538": 538,
            "id10539": 539,
            "id10547": 540,
            "id10540": 541,
            "id10541": 542,
            "id10542": 543,
            "id10543": 544,
            "id10544": 545,
            "id10545": 546,
            "id10553": 547,
            "id10554": 548,
            "id10548": 549,
            "id10549": 550,
            "id10550": 551,
            "id10551": 552,
            "id10552": 553,
            "id10556": 554,
            "id10555": 555,
            "id10557": 556,
            "id10558": 557,
            "id10559": 558,
            "id10560": 559,
            "id10561": 560,
            "id10562": 561,
            "id10563": 562,
            "id10564": 563,
            "id10565": 564,
            "id10566": 565,
            "id10567": 566,
            "id10568": 567,
            "id10569": 568,
            "id10570": 569,
            "id10571": 570,
            "id10572": 571,
            "id10576": 572,
            "id10577": 573,
            "id10575": 574,
            "id10573": 575,
            "id10574": 576,
            "id10578": 577,
            "id10579": 578,
            "id10580": 579,
            "id10581": 580,
            "id10582": 581,
            "id10583": 582,
            "id10584": 583,
            "id10586": 584,
            "id10585": 585,
            "id10587": 586,
            "id10588": 587,
            "id10589": 588,
            "id10597": 589,
            "id10590": 590,
            "id10591": 591,
            "id10592": 592,
            "id10593": 593,
            "id10594": 594,
            "id10595": 595,
            "id10596": 596,
            "id10598": 597,
            "id10599": 598,
            "id10600": 599,
            "id10601": 600,
            "id10602": 601,
            "id10603": 602,
            "id10604": 603,
            "id10605": 604,
            "id10606": 605,
            "id10607": 606,
            "id10608": 607,
            "id10609": 608,
            "id10610": 609,
            "id10611": 610,
            "id10612": 611,
            "id10613": 612,
            "id10614": 613,
            "id10615": 614,
            "id10616": 615,
            "id10617": 616,
            "id10618": 617,
            "id10619": 618,
            "id10620": 619,
            "id10622": 620,
            "id10623": 621,
            "id10624": 622,
            "id10625": 623,
            "id10626": 624,
            "id10627": 625,
            "id10628": 626,
            "id10629": 627,
            "id10630": 628,
            "id10621": 629,
            "id10633": 630,
            "id10634": 631,
            "id10631": 632,
            "id10632": 633,
            "id10635": 634,
            "id10636": 635,
            "id10637": 636,
            "id10638": 637,
            "id10639": 638,
            "id10640": 639,
            "id10641": 640,
            "id10642": 641,
            "id10643": 642,
            "id10644": 643,
            "id10645": 644,
            "id10646": 645,
            "id10647": 646,
            "id10648": 647,
            "id10651": 648,
            "id10652": 649,
            "id10649": 650,
            "id10650": 651,
            "id10653": 652,
            "id10654": 653,
            "id10655": 654,
            "id10656": 655,
            "id10657": 656,
            "id10658": 657,
            "id10659": 658,
            "id10660": 659,
            "id10661": 660,
            "id10662": 661,
            "id10663": 662,
            "id10664": 663,
            "id10665": 664,
            "id10666": 665,
            "id10667": 666,
            "id10669": 667,
            "id10668": 668,
            "id10670": 669,
            "id10671": 670,
            "id10672": 671,
            "id10673": 672,
            "id10674": 673,
            "id10675": 674,
            "id10676": 675,
            "id10677": 676,
            "id10678": 677,
            "id10679": 678,
            "id10680": 679,
            "id10681": 680,
            "id10682": 681,
            "id10683": 682,
            "id10684": 683,
            "id10685": 684,
            "id10686": 685,
            "id10687": 686,
            "id10689": 687,
            "id10690": 688,
            "id10691": 689,
            "id10692": 690,
            "id10693": 691,
            "id10694": 692,
            "id10688": 693,
            "id10695": 694,
            "id10696": 695,
            "id10697": 696,
            "id10698": 697,
            "id10699": 698,
            "id10700": 699,
            "id10701": 700,
            "id10702": 701,
            "id10703": 702,
            "id10704": 703,
            "id10705": 704,
            "id10706": 705,
            "id10707": 706,
            "id10708": 707,
            "id10709": 708,
            "id10710": 709,
            "id10711": 710,
            "id10713": 711,
            "id10714": 712,
            "id10712": 713,
            "id10715": 714,
            "id10716": 715,
            "id10717": 716,
            "id10718": 717,
            "id10719": 718,
            "id10720": 719,
            "id10721": 720,
            "id10722": 721,
            "id10723": 722,
            "id10724": 723,
            "id10725": 724,
            "id10726": 725,
            "id10727": 726,
            "id10728": 727,
            "id10729": 728,
            "id10731": 729,
            "id10732": 730,
            "id10730": 731,
            "id10733": 732,
            "id10734": 733,
            "id10735": 734,
            "id10736": 735,
            "id10740": 736,
            "id10737": 737,
            "id10738": 738,
            "id10739": 739,
            "id10741": 740,
            "id10743": 741,
            "id10742": 742,
            "id10744": 743,
            "id10745": 744,
            "id10746": 745,
            "id10747": 746,
            "id10748": 747,
            "id10749": 748,
            "id10750": 749,
            "id10751": 750,
            "id10752": 751,
            "id10753": 752,
            "id10754": 753,
            "id10755": 754,
            "id10756": 755,
            "id10757": 756,
            "id10758": 757,
            "id10759": 758,
            "id10763": 759,
            "id10760": 760,
            "id10761": 761,
            "id10762": 762,
            "id10765": 763,
            "id10764": 764,
            "id10766": 765,
            "id10767": 766,
            "id10768": 767,
            "id10769": 768,
            "id10770": 769,
            "id10771": 770,
            "id10772": 771,
            "id10775": 772,
            "id10776": 773,
            "id10777": 774,
            "id10778": 775,
            "id10779": 776,
            "id10780": 777,
            "id10773": 778,
            "id10774": 779,
            "id10781": 780,
            "id10782": 781,
            "id10783": 782,
            "id10784": 783,
            "id10785": 784,
            "id10786": 785,
            "id10787": 786,
            "id10788": 787,
            "id10789": 788,
            "id10790": 789,
            "id10791": 790,
            "id10792": 791,
            "id10793": 792,
            "id10794": 793,
            "id10795": 794,
            "id10796": 795,
            "id10797": 796,
            "id10798": 797,
            "id10799": 798,
            "id10816": 799,
            "id10800": 800,
            "id10802": 801,
            "id10801": 802,
            "id10803": 803,
            "id10804": 804,
            "id10805": 805,
            "id10806": 806,
            "id10808": 807,
            "id10807": 808,
            "id10809": 809,
            "id10810": 810,
            "id10811": 811,
            "id10812": 812,
            "id10813": 813,
            "id10814": 814,
            "id10815": 815,
            "id10817": 816,
            "id10818": 817,
            "id10819": 818,
            "id10820": 819,
            "id10821": 820,
            "id10822": 821,
            "id10823": 822,
            "id10824": 823,
            "id10825": 824,
            "id10826": 825,
            "id10827": 826,
            "id10828": 827,
            "id10829": 828,
            "id10830": 829,
            "id10831": 830,
            "id10832": 831,
            "id10833": 832,
            "id10834": 833,
            "id10835": 834,
            "id10836": 835,
            "id10837": 836,
            "id10838": 837,
            "id10839": 838,
            "id10840": 839,
            "id10841": 840,
            "id10842": 841,
            "id10843": 842,
            "id10844": 843,
            "id10845": 844,
            "id10847": 845,
            "id10846": 846,
            "id10848": 847,
            "id10849": 848,
            "id10850": 849,
            "id10851": 850,
            "id10852": 851,
            "id10853": 852,
            "id10854": 853,
            "id10855": 854,
            "id10856": 855,
            "id10857": 856,
            "id10858": 857,
            "id10859": 858,
            "id10860": 859,
            "id10861": 860,
            "id10862": 861,
            "id10863": 862,
            "id10864": 863,
            "id10866": 864,
            "id10868": 865,
            "id10867": 866,
            "id10869": 867,
            "id10870": 868,
            "id10865": 869,
            "id10871": 870,
            "id10872": 871,
            "id10873": 872,
            "id10876": 873,
            "id10874": 874,
            "id10875": 875,
            "id10877": 876,
            "id10878": 877,
            "id10879": 878,
            "id10880": 879,
            "id10881": 880,
            "id10882": 881,
            "id10883": 882,
            "id10884": 883,
            "id10885": 884,
            "id10886": 885,
            "id10887": 886,
            "id10888": 887,
            "id10889": 888,
            "id10890": 889,
            "id10891": 890,
            "id10892": 891,
            "id10893": 892,
            "id10894": 893,
            "id10895": 894,
            "id10896": 895,
            "id10897": 896,
            "id10898": 897,
            "id10899": 898,
            "id10900": 899,
            "id10902": 900,
            "id10901": 901,
            "id10903": 902,
            "id10904": 903,
            "id10905": 904,
            "id10906": 905,
            "id10907": 906,
            "id10908": 907,
            "id10909": 908,
            "id10910": 909,
            "id10911": 910,
            "id10912": 911,
            "id10913": 912,
            "id10914": 913,
            "id10915": 914,
            "id10916": 915,
            "id10917": 916,
            "id10918": 917,
            "id10919": 918,
            "id10920": 919,
            "id10921": 920,
            "id10922": 921,
            "id10923": 922,
            "id10924": 923,
            "id10925": 924,
            "id10926": 925,
            "id10927": 926,
            "id10928": 927,
            "id10929": 928,
            "id10931": 929,
            "id10932": 930,
            "id10933": 931,
            "id10934": 932,
            "id10935": 933,
            "id10936": 934,
            "id10930": 935,
            "id10937": 936,
            "id10938": 937,
            "id10939": 938,
            "id10940": 939,
            "id10941": 940,
            "id10942": 941,
            "id10943": 942,
            "id10944": 943,
            "id10945": 944,
            "id10947": 945,
            "id10948": 946,
            "id10949": 947,
            "id10950": 948,
            "id10951": 949,
            "id10952": 950,
            "id10953": 951,
            "id10954": 952,
            "id10955": 953,
            "id10956": 954,
            "id10957": 955,
            "id10958": 956,
            "id10959": 957,
            "id10960": 958,
            "id10961": 959,
            "id10962": 960,
            "id10963": 961,
            "id10964": 962,
            "id10965": 963,
            "id10966": 964,
            "id10967": 965,
            "id10968": 966,
            "id10969": 967,
            "id10970": 968,
            "id10971": 969,
            "id10972": 970,
            "id10973": 971,
            "id10974": 972,
            "id10975": 973,
            "id10976": 974,
            "id10977": 975,
            "id10978": 976,
            "id10979": 977,
            "id10980": 978,
            "id10981": 979,
            "id10982": 980,
            "id10988": 981,
            "id10989": 982,
            "id10983": 983,
            "id10984": 984,
            "id10985": 985,
            "id10990": 986,
            "id10991": 987,
            "id10992": 988,
            "id10993": 989,
            "id10994": 990,
            "id10995": 991,
            "id10986": 992,
            "id10987": 993,
            "id10996": 994,
            "id10997": 995,
            "id10998": 996,
            "id11002": 997,
            "id10999": 998,
            "id11000": 999,
            "id11001": 1000,
            "id11003": 1001,
            "id11004": 1002,
            "id11005": 1003,
            "id11006": 1004,
            "id11007": 1005,
            "id11008": 1006,
            "id11009": 1007,
            "id11010": 1008,
            "id11011": 1009,
            "id11012": 1010,
            "id11013": 1011,
            "id11014": 1012,
            "id11015": 1013,
            "id11016": 1014,
            "id11017": 1015,
            "id11018": 1016,
            "id11019": 1017,
            "id11020": 1018,
            "id11021": 1019,
            "id10946": 1020,
            "id11023": 1021,
            "id11032": 1022,
            "id11024": 1023,
            "id11033": 1024,
            "id11034": 1025,
            "id11035": 1026,
            "id11025": 1027,
            "id11026": 1028,
            "id11027": 1029,
            "id11028": 1030,
            "id11029": 1031,
            "id11036": 1032,
            "id11030": 1033,
            "id11031": 1034,
            "id11037": 1035,
            "id11038": 1036,
            "id11039": 1037,
            "id11040": 1038,
            "id11041": 1039,
            "id11043": 1040,
            "id11044": 1041,
            "id11045": 1042,
            "id11046": 1043,
            "id11047": 1044,
            "id11048": 1045,
            "id11049": 1046,
            "id11042": 1047,
            "id11050": 1048,
            "id11051": 1049,
            "id11052": 1050,
            "id11053": 1051,
            "id11054": 1052,
            "id11055": 1053,
            "id11056": 1054,
            "id11057": 1055,
            "id11060": 1056,
            "id11058": 1057,
            "id11059": 1058,
            "id11061": 1059,
            "id11062": 1060,
            "id11063": 1061,
            "id11064": 1062,
            "id11065": 1063,
            "id11022": 1064,
            "id11066": 1065,
            "id11067": 1066,
            "id11068": 1067,
            "id11069": 1068,
            "id11070": 1069,
            "id11071": 1070,
            "id11072": 1071,
            "id11073": 1072,
            "id11074": 1073,
            "id11075": 1074,
            "id11076": 1075,
            "id11077": 1076,
            "id11078": 1077,
            "id11079": 1078,
            "id11080": 1079,
            "id11081": 1080,
            "id11082": 1081,
            "id11083": 1082,
            "id11084": 1083,
            "id11085": 1084,
            "id11086": 1085,
            "id11087": 1086,
            "id11088": 1087,
            "id11089": 1088,
            "id11090": 1089,
            "id11091": 1090,
            "id11092": 1091,
            "id11093": 1092,
            "id11094": 1093,
            "id11095": 1094,
            "id11096": 1095,
            "id11097": 1096,
            "id11098": 1097,
            "id11099": 1098,
            "id11100": 1099,
            "id11101": 1100,
            "id11102": 1101,
            "id11103": 1102,
            "id11104": 1103,
            "id11106": 1104,
            "id11105": 1105,
            "id11107": 1106,
            "id11108": 1107,
            "id11109": 1108,
            "id11110": 1109,
            "id11111": 1110,
            "id11112": 1111,
            "id11113": 1112,
            "id11114": 1113,
            "id11115": 1114,
            "id11116": 1115,
            "id11117": 1116,
            "id11118": 1117,
            "id11119": 1118,
            "id11120": 1119,
            "id11121": 1120,
            "id11122": 1121,
            "id11123": 1122,
            "id11124": 1123,
            "id11125": 1124,
            "id11126": 1125,
            "id11127": 1126,
            "id11128": 1127,
            "id11129": 1128,
            "id11130": 1129,
            "id11131": 1130,
            "id11132": 1131,
            "id11133": 1132,
            "id11134": 1133,
            "id11136": 1134,
            "id11137": 1135,
            "id11138": 1136,
            "id11139": 1137,
            "id11140": 1138,
            "id11141": 1139,
            "id11143": 1140,
            "id11142": 1141,
            "id11144": 1142,
            "id11145": 1143,
            "id11146": 1144,
            "id11147": 1145,
            "id11148": 1146,
            "id11149": 1147,
            "id11150": 1148,
            "id11151": 1149,
            "id11152": 1150,
            "id11153": 1151,
            "id11154": 1152,
            "id11155": 1153,
            "id11156": 1154,
            "id11157": 1155,
            "id11158": 1156,
            "id11159": 1157,
            "id11160": 1158,
            "id11161": 1159,
            "id11162": 1160,
            "id11163": 1161,
            "id11164": 1162,
            "id11165": 1163,
            "id11166": 1164,
            "id11167": 1165,
            "id11168": 1166,
            "id11169": 1167,
            "id11170": 1168,
            "id11171": 1169,
            "id11172": 1170,
            "id11173": 1171,
            "id11174": 1172,
            "id11175": 1173,
            "id11176": 1174,
            "id11135": 1175,
            "id11177": 1176,
            "id11178": 1177,
            "id11179": 1178,
            "id11180": 1179,
            "id11181": 1180,
            "id11182": 1181,
            "id11183": 1182,
            "id11184": 1183,
            "id11185": 1184,
            "id11186": 1185,
            "id11187": 1186,
            "id11188": 1187,
            "id11189": 1188,
            "id11190": 1189,
            "id11191": 1190,
            "id11192": 1191,
            "id11193": 1192,
            "id11195": 1193,
            "id11196": 1194,
            "id11197": 1195,
            "id11198": 1196,
            "id11194": 1197,
            "id11199": 1198,
            "id11200": 1199,
            "id11201": 1200,
            "id11202": 1201,
            "id11203": 1202,
            "id11204": 1203,
            "id11205": 1204,
            "id11206": 1205,
            "id11207": 1206,
            "id11208": 1207,
            "id11209": 1208,
            "id11210": 1209,
            "id11211": 1210,
            "id11212": 1211,
            "id11213": 1212,
            "id11214": 1213,
            "id11215": 1214,
            "id11216": 1215,
            "id11217": 1216,
            "id11218": 1217,
            "id11219": 1218,
            "id11220": 1219,
            "id11221": 1220,
            "id11222": 1221,
            "id11223": 1222,
            "id11224": 1223,
            "id11225": 1224,
            "id11226": 1225,
            "id11227": 1226,
            "id11232": 1227,
            "id11228": 1228,
            "id11233": 1229,
            "id11234": 1230,
            "id11235": 1231,
            "id11229": 1232,
            "id11230": 1233,
            "id11231": 1234,
            "id11236": 1235,
            "id11237": 1236,
            "id11238": 1237,
            "id11239": 1238,
            "id11240": 1239,
            "id11241": 1240,
            "id11242": 1241,
            "id11243": 1242,
            "id11246": 1243,
            "id11247": 1244,
            "id11248": 1245,
            "id11244": 1246,
            "id11245": 1247,
            "id11249": 1248,
            "id11250": 1249,
            "id11251": 1250,
        }
        task_config = TaskConfig(
            batch_size_train=64,
            learning_rate=1e-3,
            train_split="voxceleb1_train",
            test_split="voxceleb1_test",
            valid_split="voxceleb1_valid",
            zenodo_id="TODO",
            output_dim=len(self.class_label_maps),
            epochs=50,
        )
        super().__init__(encoder, config=task_config)
        self.label_processor = lambda x: self.class_label_maps[x[data_key]]

    def run(self) -> float:
        return self.default_run()

    def make_encoded_tar(self):
        self.default_make_encoded_tar()
