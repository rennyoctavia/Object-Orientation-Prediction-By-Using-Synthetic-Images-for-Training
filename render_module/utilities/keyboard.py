
class Key:
	def __init__(self, keyCode):
		self.keyCode = keyCode
		self.isDown = False
		self.timeDown = 0
		self.timeUp = 0
class Keyboard:
	def __init__(self):
		self.pygame = None

		self.keys = []

		self.keyBackspace = None
		self.keyTab = None
		self.keyClear = None
		self.keyReturn = None
		self.keyPause = None
		self.keyEscape = None
		self.keySpace = None
		self.keyExclaim = None
		self.keyQuotedbl = None
		self.keyHash = None
		self.keyDollar = None
		self.keyAmpersand = None
		self.keyQuote = None
		self.keyLeftparen = None
		self.keyRightparen = None
		self.keyAsterisk = None
		self.keyPlus = None
		self.keyComma = None
		self.keyMinus = None
		self.keyPeriod = None
		self.keySlash = None
		self.key0 = None
		self.key1 = None
		self.key2 = None
		self.key3 = None
		self.key4 = None
		self.key5 = None
		self.key6 = None
		self.key7 = None
		self.key8 = None
		self.key9 = None
		self.keyColon = None
		self.keySemicolon = None
		self.keyLess = None
		self.keyEquals = None
		self.keyGreater = None
		self.keyQuestion = None
		self.keyAt = None
		self.keyLeftbracket = None
		self.keyBackslash = None
		self.keyRightbracket = None
		self.keyCaret = None
		self.keyUnderscore = None
		self.keyBackquote = None
		self.keyA = None
		self.keyB = None
		self.keyC = None
		self.keyD = None
		self.keyE = None
		self.keyF = None
		self.keyG = None
		self.keyH = None
		self.keyI = None
		self.keyJ = None
		self.keyK = None
		self.keyL = None
		self.keyM = None
		self.keyN = None
		self.keyO = None
		self.keyP = None
		self.keyQ = None
		self.keyR = None
		self.keyS = None
		self.keyT = None
		self.keyU = None
		self.keyV = None
		self.keyW = None
		self.keyX = None
		self.keyY = None
		self.keyZ = None
		self.keyDelete = None
		self.keyKP0 = None
		self.keyKP1 = None
		self.keyKP2 = None
		self.keyKP3 = None
		self.keyKP4 = None
		self.keyKP5 = None
		self.keyKP6 = None
		self.keyKP7 = None
		self.keyKP8 = None
		self.keyKP9 = None
		self.keyKP_period = None
		self.keyKP_divide = None
		self.keyKP_multiply = None
		self.keyKP_minus = None
		self.keyKP_plus = None
		self.keyKP_enter = None
		self.keyKP_equals = None
		self.keyUp = None
		self.keyDown = None
		self.keyRight = None
		self.keyLeft = None
		self.keyInsert = None
		self.keyHome = None
		self.keyEnd = None
		self.keyPageup = None
		self.keyPagedown = None
		self.keyF1 = None
		self.keyF2 = None
		self.keyF3 = None
		self.keyF4 = None
		self.keyF5 = None
		self.keyF6 = None
		self.keyF7 = None
		self.keyF8 = None
		self.keyF9 = None
		self.keyF10 = None
		self.keyF11 = None
		self.keyF12 = None
		self.keyF13 = None
		self.keyF14 = None
		self.keyF15 = None
		self.keyNumlock = None
		self.keyCapslock = None
		self.keyScrollock = None
		self.keyRshift = None
		self.keyLshift = None
		self.keyRctrl = None
		self.keyLctrl = None
		self.keyRalt = None
		self.keyLalt = None
		self.keyRmeta = None
		self.keyLmeta = None
		self.keyLsuper = None
		self.keyRsuper = None
		self.keyMode = None
		self.keyHelp = None
		self.keyPrint = None
		self.keySysreq = None
		self.keyBreak = None
		self.keyMenu = None
		self.keyPower = None
		self.keyEuro = None
	def setPyGame(self, pygame):
		self.pygame = pygame

		self.keyBackspace = Key(self.pygame.K_BACKSPACE)
		self.keys.append(self.keyBackspace)
		self.keyTab = Key(self.pygame.K_TAB)
		self.keys.append(self.keyTab)
		self.keyClear = Key(self.pygame.K_CLEAR)
		self.keys.append(self.keyClear)
		self.keyReturn = Key(self.pygame.K_RETURN)
		self.keys.append(self.keyReturn)
		self.keyPause = Key(self.pygame.K_PAUSE)
		self.keys.append(self.keyPause)
		self.keyEscape = Key(self.pygame.K_ESCAPE)
		self.keys.append(self.keyEscape)
		self.keySpace = Key(self.pygame.K_SPACE)
		self.keys.append(self.keySpace)
		self.keyExclaim = Key(self.pygame.K_EXCLAIM)
		self.keys.append(self.keyExclaim)
		self.keyQuotedbl = Key(self.pygame.K_QUOTEDBL)
		self.keys.append(self.keyQuotedbl)
		self.keyHash = Key(self.pygame.K_HASH)
		self.keys.append(self.keyHash)
		self.keyDollar = Key(self.pygame.K_DOLLAR)
		self.keys.append(self.keyDollar)
		self.keyAmpersand = Key(self.pygame.K_AMPERSAND)
		self.keys.append(self.keyAmpersand)
		self.keyQuote = Key(self.pygame.K_QUOTE)
		self.keys.append(self.keyQuote)
		self.keyLeftparen = Key(self.pygame.K_LEFTPAREN)
		self.keys.append(self.keyLeftparen)
		self.keyRightparen = Key(self.pygame.K_RIGHTPAREN)
		self.keys.append(self.keyRightparen)
		self.keyAsterisk = Key(self.pygame.K_ASTERISK)
		self.keys.append(self.keyAsterisk)
		self.keyPlus = Key(self.pygame.K_PLUS)
		self.keys.append(self.keyPlus)
		self.keyComma = Key(self.pygame.K_COMMA)
		self.keys.append(self.keyComma)
		self.keyMinus = Key(self.pygame.K_MINUS)
		self.keys.append(self.keyMinus)
		self.keyPeriod = Key(self.pygame.K_PERIOD)
		self.keys.append(self.keyPeriod)
		self.keySlash = Key(self.pygame.K_SLASH)
		self.keys.append(self.keySlash)
		self.key0 = Key(self.pygame.K_0)
		self.keys.append(self.key0)
		self.key1 = Key(self.pygame.K_1)
		self.keys.append(self.key1)
		self.key2 = Key(self.pygame.K_2)
		self.keys.append(self.key2)
		self.key3 = Key(self.pygame.K_3)
		self.keys.append(self.key3)
		self.key4 = Key(self.pygame.K_4)
		self.keys.append(self.key4)
		self.key5 = Key(self.pygame.K_5)
		self.keys.append(self.key5)
		self.key6 = Key(self.pygame.K_6)
		self.keys.append(self.key6)
		self.key7 = Key(self.pygame.K_7)
		self.keys.append(self.key7)
		self.key8 = Key(self.pygame.K_8)
		self.keys.append(self.key8)
		self.key9 = Key(self.pygame.K_9)
		self.keys.append(self.key9)
		self.keyColon = Key(self.pygame.K_COLON)
		self.keys.append(self.keyColon)
		self.keySemicolon = Key(self.pygame.K_SEMICOLON)
		self.keys.append(self.keySemicolon)
		self.keyLess = Key(self.pygame.K_LESS)
		self.keys.append(self.keyLess)
		self.keyEquals = Key(self.pygame.K_EQUALS)
		self.keys.append(self.keyEquals)
		self.keyGreater = Key(self.pygame.K_GREATER)
		self.keys.append(self.keyGreater)
		self.keyQuestion = Key(self.pygame.K_QUESTION)
		self.keys.append(self.keyQuestion)
		self.keyAt = Key(self.pygame.K_AT)
		self.keys.append(self.keyAt)
		self.keyLeftbracket = Key(self.pygame.K_LEFTBRACKET)
		self.keys.append(self.keyLeftbracket)
		self.keyBackslash = Key(self.pygame.K_BACKSLASH)
		self.keys.append(self.keyBackslash)
		self.keyRightbracket = Key(self.pygame.K_RIGHTBRACKET)
		self.keys.append(self.keyRightbracket)
		self.keyCaret = Key(self.pygame.K_CARET)
		self.keys.append(self.keyCaret)
		self.keyUnderscore = Key(self.pygame.K_UNDERSCORE)
		self.keys.append(self.keyUnderscore)
		self.keyBackquote = Key(self.pygame.K_BACKQUOTE)
		self.keys.append(self.keyBackquote)
		self.keyA = Key(self.pygame.K_a)
		self.keys.append(self.keyA)
		self.keyB = Key(self.pygame.K_b)
		self.keys.append(self.keyB)
		self.keyC = Key(self.pygame.K_c)
		self.keys.append(self.keyC)
		self.keyD = Key(self.pygame.K_d)
		self.keys.append(self.keyD)
		self.keyE = Key(self.pygame.K_e)
		self.keys.append(self.keyE)
		self.keyF = Key(self.pygame.K_f)
		self.keys.append(self.keyF)
		self.keyG = Key(self.pygame.K_g)
		self.keys.append(self.keyG)
		self.keyH = Key(self.pygame.K_h)
		self.keys.append(self.keyH)
		self.keyI = Key(self.pygame.K_i)
		self.keys.append(self.keyI)
		self.keyJ = Key(self.pygame.K_j)
		self.keys.append(self.keyJ)
		self.keyK = Key(self.pygame.K_k)
		self.keys.append(self.keyK)
		self.keyL = Key(self.pygame.K_l)
		self.keys.append(self.keyL)
		self.keyM = Key(self.pygame.K_m)
		self.keys.append(self.keyM)
		self.keyN = Key(self.pygame.K_n)
		self.keys.append(self.keyN)
		self.keyO = Key(self.pygame.K_o)
		self.keys.append(self.keyO)
		self.keyP = Key(self.pygame.K_p)
		self.keys.append(self.keyP)
		self.keyQ = Key(self.pygame.K_q)
		self.keys.append(self.keyQ)
		self.keyR = Key(self.pygame.K_r)
		self.keys.append(self.keyR)
		self.keyS = Key(self.pygame.K_s)
		self.keys.append(self.keyS)
		self.keyT = Key(self.pygame.K_t)
		self.keys.append(self.keyT)
		self.keyU = Key(self.pygame.K_u)
		self.keys.append(self.keyU)
		self.keyV = Key(self.pygame.K_v)
		self.keys.append(self.keyV)
		self.keyW = Key(self.pygame.K_w)
		self.keys.append(self.keyW)
		self.keyX = Key(self.pygame.K_x)
		self.keys.append(self.keyX)
		self.keyY = Key(self.pygame.K_y)
		self.keys.append(self.keyY)
		self.keyZ = Key(self.pygame.K_z)
		self.keys.append(self.keyZ)
		self.keyDelete = Key(self.pygame.K_DELETE)
		self.keys.append(self.keyDelete)
		self.keyKP0 = Key(self.pygame.K_KP0)
		self.keys.append(self.keyKP0)
		self.keyKP1 = Key(self.pygame.K_KP1)
		self.keys.append(self.keyKP1)
		self.keyKP2 = Key(self.pygame.K_KP2)
		self.keys.append(self.keyKP2)
		self.keyKP3 = Key(self.pygame.K_KP3)
		self.keys.append(self.keyKP3)
		self.keyKP4 = Key(self.pygame.K_KP4)
		self.keys.append(self.keyKP4)
		self.keyKP5 = Key(self.pygame.K_KP5)
		self.keys.append(self.keyKP5)
		self.keyKP6 = Key(self.pygame.K_KP6)
		self.keys.append(self.keyKP6)
		self.keyKP7 = Key(self.pygame.K_KP7)
		self.keys.append(self.keyKP7)
		self.keyKP8 = Key(self.pygame.K_KP8)
		self.keys.append(self.keyKP8)
		self.keyKP9 = Key(self.pygame.K_KP9)
		self.keys.append(self.keyKP9)
		self.keyKP_period = Key(self.pygame.K_KP_PERIOD)
		self.keys.append(self.keyKP_period)
		self.keyKP_divide = Key(self.pygame.K_KP_DIVIDE)
		self.keys.append(self.keyKP_divide)
		self.keyKP_multiply = Key(self.pygame.K_KP_MULTIPLY)
		self.keys.append(self.keyKP_multiply)
		self.keyKP_minus = Key(self.pygame.K_KP_MINUS)
		self.keys.append(self.keyKP_minus)
		self.keyKP_plus = Key(self.pygame.K_KP_PLUS)
		self.keys.append(self.keyKP_plus)
		self.keyKP_enter = Key(self.pygame.K_KP_ENTER)
		self.keys.append(self.keyKP_enter)
		self.keyKP_equals = Key(self.pygame.K_KP_EQUALS)
		self.keys.append(self.keyKP_equals)
		self.keyUp = Key(self.pygame.K_UP)
		self.keys.append(self.keyUp)
		self.keyDown = Key(self.pygame.K_DOWN)
		self.keys.append(self.keyDown)
		self.keyRight = Key(self.pygame.K_RIGHT)
		self.keys.append(self.keyRight)
		self.keyLeft = Key(self.pygame.K_LEFT)
		self.keys.append(self.keyLeft)
		self.keyInsert = Key(self.pygame.K_INSERT)
		self.keys.append(self.keyInsert)
		self.keyHome = Key(self.pygame.K_HOME)
		self.keys.append(self.keyHome)
		self.keyEnd = Key(self.pygame.K_END)
		self.keys.append(self.keyEnd)
		self.keyPageup = Key(self.pygame.K_PAGEUP)
		self.keys.append(self.keyPageup)
		self.keyPagedown = Key(self.pygame.K_PAGEDOWN)
		self.keys.append(self.keyPagedown)
		self.keyF1 = Key(self.pygame.K_F1)
		self.keys.append(self.keyF1)
		self.keyF2 = Key(self.pygame.K_F2)
		self.keys.append(self.keyF2)
		self.keyF3 = Key(self.pygame.K_F3)
		self.keys.append(self.keyF3)
		self.keyF4 = Key(self.pygame.K_F4)
		self.keys.append(self.keyF4)
		self.keyF5 = Key(self.pygame.K_F5)
		self.keys.append(self.keyF5)
		self.keyF6 = Key(self.pygame.K_F6)
		self.keys.append(self.keyF6)
		self.keyF7 = Key(self.pygame.K_F7)
		self.keys.append(self.keyF7)
		self.keyF8 = Key(self.pygame.K_F8)
		self.keys.append(self.keyF8)
		self.keyF9 = Key(self.pygame.K_F9)
		self.keys.append(self.keyF9)
		self.keyF10 = Key(self.pygame.K_F10)
		self.keys.append(self.keyF10)
		self.keyF11 = Key(self.pygame.K_F11)
		self.keys.append(self.keyF11)
		self.keyF12 = Key(self.pygame.K_F12)
		self.keys.append(self.keyF12)
		self.keyF13 = Key(self.pygame.K_F13)
		self.keys.append(self.keyF13)
		self.keyF14 = Key(self.pygame.K_F14)
		self.keys.append(self.keyF14)
		self.keyF15 = Key(self.pygame.K_F15)
		self.keys.append(self.keyF15)
		self.keyNumlock = Key(self.pygame.K_NUMLOCK)
		self.keys.append(self.keyNumlock)
		self.keyCapslock = Key(self.pygame.K_CAPSLOCK)
		self.keys.append(self.keyCapslock)
		self.keyScrollock = Key(self.pygame.K_SCROLLOCK)
		self.keys.append(self.keyScrollock)
		self.keyRshift = Key(self.pygame.K_RSHIFT)
		self.keys.append(self.keyRshift)
		self.keyLshift = Key(self.pygame.K_LSHIFT)
		self.keys.append(self.keyLshift)
		self.keyRctrl = Key(self.pygame.K_RCTRL)
		self.keys.append(self.keyRctrl)
		self.keyLctrl = Key(self.pygame.K_LCTRL)
		self.keys.append(self.keyLctrl)
		self.keyRalt = Key(self.pygame.K_RALT)
		self.keys.append(self.keyRalt)
		self.keyLalt = Key(self.pygame.K_LALT)
		self.keys.append(self.keyLalt)
		self.keyRmeta = Key(self.pygame.K_RMETA)
		self.keys.append(self.keyRmeta)
		self.keyLmeta = Key(self.pygame.K_LMETA)
		self.keys.append(self.keyLmeta)
		self.keyLsuper = Key(self.pygame.K_LSUPER)
		self.keys.append(self.keyLsuper)
		self.keyRsuper = Key(self.pygame.K_RSUPER)
		self.keys.append(self.keyRsuper)
		self.keyMode = Key(self.pygame.K_MODE)
		self.keys.append(self.keyMode)
		self.keyHelp = Key(self.pygame.K_HELP)
		self.keys.append(self.keyHelp)
		self.keyPrint = Key(self.pygame.K_PRINT)
		self.keys.append(self.keyPrint)
		self.keySysreq = Key(self.pygame.K_SYSREQ)
		self.keys.append(self.keySysreq)
		self.keyBreak = Key(self.pygame.K_BREAK)
		self.keys.append(self.keyBreak)
		self.keyMenu = Key(self.pygame.K_MENU)
		self.keys.append(self.keyMenu)
		self.keyPower = Key(self.pygame.K_POWER)
		self.keys.append(self.keyPower)
		self.keyEuro = Key(self.pygame.K_EURO)
		self.keys.append(self.keyEuro)
	def update(self, events):
		#
		for event in events:
			if event.type == self.pygame.KEYDOWN:
				for key in self.keys:
					if event.key == key.keyCode:
						if not key.isDown:
							key.timeDown = 0
							key.isDown = True
			if event.type == self.pygame.KEYUP:
				for key in self.keys:
					if event.key == key.keyCode:
						if key.isDown:
							key.timeUp = 0
							key.isDown = False
		#
		for key in self.keys:
			if key.isDown:
				key.timeDown += 1
			else:
				key.timeUp += 1



