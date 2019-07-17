#include <algorithm>
#include <math.h>
#include <iomanip>
#include <mpi.h>
#include <string>
#include <iostream>

#include <allegro5/allegro.h>
#include <allegro5/allegro_acodec.h>
#include <allegro5/allegro_audio.h>
#include <allegro5/allegro_font.h> 
#include <allegro5/allegro_ttf.h>
#include <allegro5/allegro_image.h>

using namespace std;

const int N = 10;        //numero di equazioni
double Matrice[N][N + 1];
//La matrice è formata dal sistema formato dai coefficienti del sistema e dal vettore b dei dati
//La N-esima colonna sono i valori di b

double MatriceOriginale[N][N + 1]; //salvo la matrice per testing successivi
double x[N] = {0}; //la soluzione

ALLEGRO_SAMPLE* up;
ALLEGRO_SAMPLE* prefinal;
ALLEGRO_SAMPLE* final;


void stampaMatrice();
void randomizzaMatrice();
bool testaSoluzione();
void checkInizializza(bool,string);
void AllegroInit();
void visualizza(int,ALLEGRO_FONT*,ALLEGRO_FONT*, ALLEGRO_DISPLAY*,double,ALLEGRO_BITMAP*);


int main(int argc, char ** argv) 
{
   int id, nproc, id_from, id_to;
   MPI_Status status;
   MPI_Init( & argc, & argv);
   MPI_Comm_rank(MPI_COMM_WORLD, & id);
   MPI_Comm_size(MPI_COMM_WORLD, & nproc);

   int rigaI, rigaJ, i, j, jmax;
   double t, amul, elem, elem_max=0;
   time_t tempoIniziale, tempoFinale;
   AllegroInit();
	ALLEGRO_FONT* font;
	ALLEGRO_FONT* fontdue;
	ALLEGRO_DISPLAY* display;
   ALLEGRO_BITMAP* sfondo;
   if (id == 0) 
	{
		font=al_load_ttf_font("arial.ttf", 30, 0);
		fontdue=al_load_ttf_font("arial.ttf", 20, 0);
		display = al_create_display(1280,720);
      al_reserve_samples(3);
      up= al_load_sample("1up.wav");
      final= al_load_sample("final.wav");
      prefinal= al_load_sample("prefinal.wav");
      sfondo=al_load_bitmap("sfondo.png");
		al_hide_mouse_cursor(display);
		visualizza(0,font,fontdue,display,tempoIniziale,sfondo); 	//L'ultimo parametro, il tempo di esecuzione ci servirà solo alla fine
		randomizzaMatrice();
		visualizza(1,font,fontdue,display,tempoIniziale,sfondo);
      if (N<=20)
         stampaMatrice();
      tempoIniziale = MPI_Wtime();
   }
	MPI_Barrier(MPI_COMM_WORLD);		// ci assicuriamo che da qui in poi la matrice sia inizializzata
   MPI_Bcast(Matrice, N * (N + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);

   for (rigaI = 0; rigaI < N - 1; ++rigaI) 
	{

      //il thread che deve eliminare la prossima riga
      //fa il broadcast a tutti gli altri
      id_from = (N - 1 - rigaI) % nproc;

      //pivoting step
      jmax = rigaI;
      elem_max = fabs(Matrice[rigaI][rigaI]);
      MPI_Bcast( & elem_max, 1, MPI_DOUBLE, id_from, MPI_COMM_WORLD);

      for (rigaJ = N - 1 - id; rigaJ >= rigaI + 1; rigaJ -= nproc)
         if (fabs(Matrice[rigaJ][rigaI]) > elem_max) 
			{
            jmax = rigaJ;
            elem_max = fabs(Matrice[rigaJ][rigaI]);
         }
      //scambia le righe se necessario
      struct 
		{
         double elem;
         int riga;
      }
      p, tmp_p;
      p.elem = elem_max;
      p.riga = jmax;
      MPI_Allreduce( & p, & tmp_p, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
      jmax = tmp_p.riga;
      //il rank dell'altro slave con cui il pivot deve essere scambiato
      id_to = (N - 1 - jmax) % nproc;
		
      if (id_from == id_to) 
		{ //sostituizione con lo stesso slave senza comunicazione
         if (id == id_from && jmax != rigaI)
            for (j = 0; j < N + 1; j++) 
				{
               elem = Matrice[rigaI][j];
               Matrice[rigaI][j] = Matrice[jmax][j];
               Matrice[jmax][j] = elem;
            }
      } 
		else 
		{ //per lo scambio di pivot ci vuole comunicazione
         if (id == id_to) 
			{
            for (j = rigaI; j < N + 1; j++)
               Matrice[rigaI][j] = Matrice[jmax][j];
            MPI_Send( & Matrice[rigaI], N + 1, MPI_DOUBLE, id_from, 77, MPI_COMM_WORLD);
            MPI_Recv( & Matrice[jmax], N + 1, MPI_DOUBLE, id_from, 88, MPI_COMM_WORLD, & status);
         }
         if (id == id_from) 
			{
            for (j = rigaI; j < N + 1; j++)
               Matrice[jmax][j] = Matrice[rigaI][j];
            MPI_Recv( & Matrice[rigaI], N + 1, MPI_DOUBLE, id_to, 77, MPI_COMM_WORLD, & status);
            MPI_Send( & Matrice[jmax], N + 1, MPI_DOUBLE, id_to, 88, MPI_COMM_WORLD);
         }
      }

      MPI_Bcast( & Matrice[rigaI], N + 1, MPI_DOUBLE, id_from, MPI_COMM_WORLD);

      //eliminazione
      t = -1.0 / Matrice[rigaI][rigaI];
      for (rigaJ = N - 1 - id; rigaJ >= rigaI + 1; rigaJ -= nproc) 
		{
         amul = Matrice[rigaJ][rigaI] * t;
         //eliminazione della riga
         for (j = rigaI; j < N + 1; j++)
            Matrice[rigaJ][j] += amul * Matrice[rigaI][j];
      }
		if (id==0)
		{
			visualizza(2,font,fontdue,display,tempoIniziale,sfondo); //visualizzazione allegro
		}
   }

   if (id == 0) 
	{
      //sostituzione finale
      for (rigaI = N - 1; rigaI >= 0; --rigaI) 
		{
         x[rigaI] = -Matrice[rigaI][N] / Matrice[rigaI][rigaI];
         for (rigaJ = 0; rigaJ < rigaI; ++rigaJ) 
			{
            Matrice[rigaJ][N] += x[rigaI] * Matrice[rigaJ][rigaI];
            Matrice[rigaJ][rigaI] = 0;
         }
      }
      tempoFinale = MPI_Wtime();
      cout << "tempo di esecuzione con "<<nproc<<" thread : "<< tempoFinale - tempoIniziale <<" secondi. n= "<<N<<endl;

		visualizza(3,font,fontdue,display,tempoFinale - tempoIniziale,sfondo);
		
      if (testaSoluzione())
		   visualizza(4,font,fontdue,display,tempoFinale - tempoIniziale,sfondo);
   }
	

	if (id==0)		//Distrugge
	{
		al_destroy_bitmap(sfondo);
      al_destroy_font(font);
		al_destroy_font(fontdue);
		al_destroy_display(display);
	}

   MPI_Finalize();
}

void randomizzaMatrice() 
{
   srand(time(NULL));
   for (int i = 0; i < N; i++)
      for (int j = 0; j < N + 1; j++)
         MatriceOriginale[i][j] = Matrice[i][j] = rand() % 10;
}

void stampaMatrice() 
{
   //Precisione su stampa std settata ad 1 elemento dopo la virgola
   cout.precision(1);
   int j=0;
   for (int i = 0; i < N + 1; i++) cout << "------";
   cout << fixed << "-----" << endl;
   for (int i = 0; i < N; i++) 
	{
      cout << "| ";
      for (j = 0; j < N; j++)
         cout << setw(5) << Matrice[i][j] << " ";
      cout << "| " << setw(5) << Matrice[i][j];
      cout << " |  x[" << setw(2) << i << "] = " << setw(5) << x[i] << endl;
   }
   for (int i = 0; i < N + 1; i++) cout << "------";
   cout << "-----" << endl;
}

bool testaSoluzione() 
{
   double diff, sum;
   cout.precision(20);
   for (int i = 0; i < N; i++) 
	{
      sum = 0;
      for (int j = 0; j < N; j++)
         sum += x[j] * MatriceOriginale[i][j];
      diff = sum + MatriceOriginale[i][N];
      if (diff > 0.0001 || diff < -0.0001)
      {
         cout << "ERRORE! " << sum << " ~ " << MatriceOriginale[i][N] << ", differenza:" << diff << endl;
         return false;
      }
      if (N < 50) 
		{
         cout << setw(4) << sum << " ~ " << setw(4) << MatriceOriginale[i][N];
         cout << ", differenza:" << setw(4) << fixed << diff << endl;
      }
   }
   return true;
}
void checkInizializza(bool test, string description)
    {
        if(test) 
            return;

        cout<<"ALLEGRO ERROR: Non ho potuto inizializzare: "<<description;
        exit(1);
    }
void AllegroInit()
{
	checkInizializza(al_init(), "allegro");
	checkInizializza(al_install_keyboard(), "keyboard");
	checkInizializza(al_install_audio(), "sound");
	checkInizializza(al_init_acodec_addon(), "Codec");
	checkInizializza(al_init_font_addon(),"font");
	checkInizializza(al_init_ttf_addon(), "ttf");
   checkInizializza(al_init_image_addon(),"image");
}
void visualizza(int stato,ALLEGRO_FONT* font,ALLEGRO_FONT* fontdue, ALLEGRO_DISPLAY* display,double esecuzione,ALLEGRO_BITMAP* sfondo)
{
   ALLEGRO_TIMER* timer = al_create_timer(1.0 / 30.0);
   checkInizializza(timer, "timer");
   ALLEGRO_EVENT_QUEUE* coda = al_create_event_queue();
   checkInizializza(coda, "coda");
   al_register_event_source(coda, al_get_keyboard_event_source());
   al_register_event_source(coda, al_get_display_event_source(display));
   al_register_event_source(coda, al_get_timer_event_source(timer));
   ALLEGRO_EVENT evento;
   al_clear_to_color(al_map_rgb(0,0,0));
	al_draw_bitmap(sfondo,0,0,0);
	switch (stato)
	{
	case 0:
      al_draw_textf(font, al_map_rgb(229, 27, 27), 0, 0, 0, "PREMI INVIO PER RANDOMIZZARE (vedere pdf per dati personalizzati)");
		break;
	case 1:
      al_draw_textf(font, al_map_rgb(161, 11, 11), 0, 0, 0, "PREMI INVIO PER FAR PARTIRE L'ALGORITMO");      
		break;
	case 2:
      al_draw_textf(font, al_map_rgb(218, 187, 36), 0, 0, 0, "STO CALCOLANDO");      
      al_play_sample(up, 1.0, 0.0,1.0,ALLEGRO_PLAYMODE_ONCE,NULL);
		break;
	case 3:
      al_draw_textf(font, al_map_rgb(36, 218, 73), 0, 0, 0, "SOLUZIONE TROVATA");
      al_draw_textf(font, al_map_rgb(36, 218, 73), 350, 0, 0, "TEMPO DI ESECUZIONE: %1.2f SECONDI. INVIO PER TESTARLA",esecuzione);
		for (int i=0;i<N;i++)
			al_draw_textf(fontdue,al_map_rgb(255, 255, 255), 100,50+i*30,0,"X %d = %1.10f",i,x[i]);
      al_play_sample(prefinal, 1.0, 0.0,1.0,ALLEGRO_PLAYMODE_ONCE,NULL);
		break;
   case 4:
      al_draw_textf(font, al_map_rgb(21, 249, 66), 0, 0, 0, "SOLUZIONE TESTATA ED ESATTA");
      al_draw_textf(font, al_map_rgb(21, 249, 66), 350, 100, 0, "TEMPO DI ESECUZIONE: %1.2f SECONDI. INVIO PER USCIRE",esecuzione);
		for (int i=0;i<N;i++)
			al_draw_textf(fontdue,al_map_rgb(255, 255, 255), 100,50+i*30,0,"X %d = %1.10f",i,x[i]);    
      al_play_sample(final, 1.0, 0.0,1.0,ALLEGRO_PLAYMODE_ONCE,NULL);        
		break;
	}
	if (stato!=3 and stato!=4)			//nella schermata finale non c'è la matrice
	{
   	for (int i=0;i<N;i++)
   	   for (int j=0;j<N+1;j++)
			{
   	   	if (N>20)
      			al_draw_textf(font, al_map_rgb(255, 255, 255), 100, 500, 0, "MATRICE TROPPO GRANDE, IMMETTI UNA DIMENSIONE MINORE DI 20 PER VISUALIZZARLA");      
				else
					al_draw_textf(fontdue, al_map_rgb(255, 255, 255), (j*65), (i*34)+30, 0, "%1.2f",Matrice[i][j]);
			}
	}
   al_flip_display();
   if (stato!=2)
   {    
		while (true)
      {
         al_wait_for_event(coda, &evento);
         if (evento.keyboard.keycode==ALLEGRO_KEY_ENTER)
            break;
			if (evento.type==ALLEGRO_EVENT_DISPLAY_CLOSE)
				exit(0);
      }
   }      
   al_flush_event_queue(coda);
   al_rest(0.8);                       //QUESTA FUNZIONE SERVE SOLO A RALLENTARE E A FAR VEDERE MEGLIO IL FUNZIONAMENTO DELL'ALGORITMO

}
